import random
from gemmforge import DenseMatrix, GenerationError, GemmGenerator, SparseMatrix
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType
from gemmforge.vm import vm_factory
import numpy as np
import sys
from random import randint
from numba import cuda
import os

from params import *

cur_dir = os.getcwd()

# b_matrix_types = ["band", "single_column_b", "single_row_b", "chequered", "full"]
a_matrix_types = ["full", "random"]


def get_available_mem_on_gpu():
    gpus = cuda.gpus.lst

    # for gpu in gpus:
    gpu = gpus[0]
    meminfo = cuda.current_context().get_memory_info()
    # print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
    return meminfo[0]


def get_suggested_num_elements(MatBSize, MatADenseSize, MatASparseSize, MatCSize, SizeOfFloat):
    # We mul A x BD(dense) = C1, A x BS(Sparse) = C2
    # And compare C1 and C1, C1 and 2 obtained back will be R1 and R2 on host
    # On host we need A, BD, BS, C, R1, R2
    # On device we need A, BD, BS, C1, C2
    per_el_size = (MatBSize + MatADenseSize +
                   MatASparseSize + MatCSize * 2) * SizeOfFloat

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    at80 = int(0.8 * can_fit_els)
    # print(f"Can fit {can_fit_els} matrices of given sizes, at 80% capacity {at80}")
    return (can_fit_els, at80)
    # return (1,1)


def gen_matrix_a(rowA, colA, transposed, atype):
    A = np.zeros([rowA, colA])
    coo = {"name": "A", "rows": rowA, "cols": colA,
           "entries": [], "coordinates": []}
    if atype == "full":
        iter = int(0)
        if transposed:
            for j in range(colA):
                for i in range(rowA):
                    coo["entries"].append([i, j, iter + 1 + colA * i])
                    coo["coordinates"].append([i, j])
                    A[i, j] = iter + 1 + colA * i
                iter += 1
        else:
            for j in range(colA):
                for i in range(rowA):
                    coo["entries"].append([i, j, iter + 1])
                    coo["coordinates"].append([i, j])
                    A[i, j] = iter + 1
                    iter += 1
        iter = 0
        a_el_count = len(coo["coordinates"])
    elif atype == "band":
        raise Exception("NO")
    elif atype == "random":
        entry_count = int(0.15 * rowB * colB)
        a_el_count = entry_count
        l = set()
        while len(l) < entry_count:
            i = randint(0, rowB - 1)
            j = randint(0, colB - 1)
            l.add((i, j))
        llist = list(l)
        assert (len(llist) == a_el_count)
        for (row, col) in llist:
            A[row, col] = 1
        for j in range(colA):
            for i in range(rowA):
                if A[i, j] != 0:
                    r = random.randint(1, 9)
                    coo["coordinates"].append([i, j])
                    coo["entries"].append([i, j, r])
                    A[i, j] = r
    else:
        raise Exception("NO")

    if transposed:
        Ao = A
        A = A.flatten("F")
    else:
        Ao = A
        A = A.flatten("F")
    T = "T"
    NT = ""
    # print(btype, f"{T if transposed else NT}: ", coo["coordinates"])
    # print(btype, f"{T if transposed else NT}: ", Ao)
    A_nonzeros = []
    for el in A:
        if el > 0.0001 or el < -0.0001:
            assert (el != 0 and el != 0.0)
            A_nonzeros.append(el)
    # print(Ao)
    # print(A)
    # print(atype, f"{T if transposed else NT} sparse: ", A_nonzeros)
    return (coo, A, A_nonzeros, a_el_count)


try:
    for with_compile_time_values in [False, True]:
        for a_type in a_matrix_types:
            for tA in [True, False]:
                for tB in [False, True]:
                    testid = ""
                    if tA:
                        testid += "At_mul_"
                    else:
                        testid += "A_mul_"
                    if tB:
                        testid += "Bt"
                    else:
                        testid += "B"
                    testid += "_" + a_type
                    valid = "_compile_time_value" if with_compile_time_values else ""
                    testid += valid

                    if not tA:
                        rowA = row_a
                        colA = col_a
                    else:
                        rowA = col_a
                        colA = row_a
                    if not tB:
                        rowB = row_b
                        colB = col_b
                    else:
                        rowB = row_b
                        colB = col_b
                    # if not tA:
                    rowC = row_c
                    colC = col_c

                    coo, matrix_a, matrix_a_non_zeros_flat, a_el_count = gen_matrix_a(
                        rowA, colA, tA, a_type)

                    mat_a_dense = DenseMatrix(num_rows=rowA,
                                              num_cols=colA,
                                              addressing="strided",
                                              bbox=[0, 0, rowA, colA])

                    mat_a_sparse = SparseMatrix(num_rows=rowA,
                                                num_cols=colA,
                                                addressing="strided",
                                                coordinates=coo["coordinates"],
                                                values=matrix_a_non_zeros_flat if with_compile_time_values else None)

                    mat_b = DenseMatrix(num_rows=rowB,
                                        num_cols=colB,
                                        bbox=[0, 0, rowB, colB],
                                        addressing="strided")

                    mat_c = DenseMatrix(num_rows=rowC,
                                        num_cols=colC,
                                        bbox=[0, 0, rowC, colC],
                                        addressing="strided")

                    vm = vm_factory(
                        arch="sm_86", backend="cuda", fp_type="float")

                    if tA:
                        transA = "Transposed"
                    else:
                        transA = ""
                    if tB:
                        transB = "Transposed"
                    else:
                        transB = ""

                    # , kernel_type=GemmKernelType.REGISTER_ONLY_BASED
                    T = "t"
                    NT = ""
                    dense_gen = GemmGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    dense_gen.set(tA, tB, mat_a_dense, mat_b, mat_c, alpha=Alpha, beta=Beta,
                                  base_name=f"A{T if transA else NT}_B{T if transB else NT}_DenseXDense")
                    dense_gen.generate()
                    # print(dense_gen.get_kernel())
                    # print(dense_gen.get_launcher())
                    # print(dense_gen.get_launcher_header())
                    dense_header = dense_gen.get_launcher_header()
                    # Get the function name without void in the beginning
                    dense_function_name = dense_header.split("(")[0][4:]

                    # , kernel_type=GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_FULL_UNIT_VECTOR_BASED
                    sparse_gen = GemmGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    sparse_gen.set(tA, tB, mat_a_sparse, mat_b, mat_c, alpha=Alpha, beta=Beta,
                                   base_name=f"A{T if transA else NT}_{a_type}_B{T if transB else NT}_SparseXDense")
                    sparse_gen.generate()
                    # print(sparse_gen.get_kernel())
                    # print(sparse_gen.get_launcher())
                    # print(sparse_gen.get_launcher_header())
                    sparse_header = sparse_gen.get_launcher_header()
                    # Get the function name without void in the beginning
                    sparse_function_name = sparse_header.split("(")[0][4:]

                    # A = np.random.random({rowA} * 9)
                    # B = np.random.random(9 * 9)
                    C = np.zeros(rowC * colC)
                    C.fill(0.1)
                    for i in range(rowC * colC):
                        C[i] = i * 0.1
                    B = np.zeros(rowB * colB)
                    B.fill(1.0)
                    for i in range(rowB * colB):
                        B[i] = i * 2.0
                    A_dense = matrix_a
                    A_sparse = matrix_a_non_zeros_flat

                    np.set_printoptions(threshold=sys.maxsize)
                    strB = np.array2string(B, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strA_sparse = np.array2string(np.array(A_sparse), separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strA_dense = np.array2string(A_dense, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ').replace(
                        "[", "{").replace("]", "}")

                    get_available_mem_on_gpu()
                    full, at80 = get_suggested_num_elements(
                        rowA * colA, rowB * colB, a_el_count, rowC * colC, 4)
                    # num_els = at80
                    num_els = 1

                    ctv = "_ctv" if with_compile_time_values else ""
                    n = f"A{T if transA else NT}_{a_type}_B{T if transB else NT}_SparseXDense{ctv}"
                    s = f"""
    #include <iostream>
    #include <cuda_runtime.h>
    #include <cstring>

    #define CHECK_ERR checkErr(__FILE__,__LINE__)

    #define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
    template <typename T>
    void check(T err, const char* const func, const char* const file,
            const int line)
    {{
        if (err != cudaSuccess)
        {{
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            // We don't exit when we encounter CUDA errors in this example.
            // std::exit(EXIT_FAILURE);
        }}
    }}

    std::string PrevFile = "";
    int PrevLine = 0;

    void checkErr(const std::string &File, int Line) {{
    #ifndef NDEBUG
        cudaError_t Error = cudaGetLastError();
        if (Error != cudaSuccess) {{
            std::cout << std::endl << File
                    << ", line " << Line
                    << ": " << cudaGetErrorString(Error)
                    << " (" << Error << ")"
                    << std::endl;

            if (PrevLine > 0)
            std::cout << "Previous CUDA call:" << std::endl
                        << PrevFile << ", line " << PrevLine << std::endl;
            throw;
        }}
        PrevFile = File;
        PrevLine = Line;
    #endif
    }}

    // Dense x Dense Kernel
    {dense_gen.get_kernel()}

    // Dense x Sparse Kernel
    {sparse_gen.get_kernel()}

    // Dense x Dense Kernel Launcher
    {dense_gen.get_launcher()}

    // Dense x Sparse Kernel Launcher
    {sparse_gen.get_launcher()}

    int main(){{
    std::cout << "Gemm-Type: " << "{n}" << std::endl;
    std::cout << "Number of elements: " << "{num_els}" << std::endl;
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;
    float CoreA_sparse[{a_el_count}] = {strA_sparse};
    float CoreA_dense[{rowA}*{colA}] = {strA_dense};
    float CoreB[{rowB} * {colB}] = {strB};
    float CoreC[{rowC}*{colC}] = {strC};
    
    // Buffers 
    std::cout << "Instantiating buffer matrices" << std::endl;
    float* A_dense = new float[{rowA}*{colA}*{num_els}];
    float* B = new float[{rowB}*{colB}*{num_els}];
    {f"float* A_sparse = new float[{a_el_count}*{num_els}];" if not with_compile_time_values else ""}
    float* C = new float[{rowC}*{colC}*{num_els}];
    float* R1 = new float[{rowC}*{colC}*{num_els}];
    float* R2 = new float[{rowC}*{colC}*{num_els}];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < {num_els}; i++){{
        std::memcpy(&A_dense[{rowA} * {colA} * i], &CoreA_dense[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&B[{rowB} * {colB} * i], &CoreB[0], {rowB} * {colB} * sizeof(float));
        {f"std::memcpy(&A_sparse[{a_el_count} * i], &CoreA_sparse[0], {a_el_count} * sizeof(float));" if not with_compile_time_values else ""}
        std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
    }}

    float *A_dense_dev = nullptr;
    {"float *A_sparse_dev = nullptr;" if not with_compile_time_values else ""}
    float *B_dev = nullptr;
    float *C1_dev = nullptr;
    float *C2_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&B_dev, sizeof(float) * {rowB} * {colB} * {num_els}); CHECK_ERR;
    {f"cudaMalloc((void **)&A_sparse_dev, sizeof(float) * {a_el_count} * {num_els}); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMalloc((void **)&A_dense_dev, sizeof(float) * {rowA} * {colA} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;

    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {rowB} * {colB} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    {f"cudaMemcpy((void *)A_sparse_dev, (void *)A_sparse, sizeof(float) *  {a_el_count} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMemcpy((void *)A_dense_dev, (void *)A_dense, sizeof(float) *  {rowA} * {colA} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;

    // Dense x Dense Matrix Mult
    std::cout << "Calling Dense x Dense kernel" << std::endl;
    float elapsedTime = 0.0; 
    cudaEvent_t startDD, stopDD;
    cudaEventCreate(&startDD); CHECK_ERR;
    cudaEventCreate(&stopDD); CHECK_ERR;
    cudaEventRecord(startDD); CHECK_ERR;
    {dense_function_name}(A_dense_dev, 0, B_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr); CHECK_ERR;
    cudaEventRecord(stopDD); CHECK_ERR;
    cudaEventSynchronize(stopDD); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime, startDD, stopDD); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    std::cout << "Dense x Dense kernel took " << elapsedTime << " ms" << std::endl; 
    cudaMemcpy(R1, C1_dev, sizeof(float)*{rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    // Sparse x Dense Matrix Mult
    std::cout << "Calling Densse x Sparse kernel" << std::endl;
    elapsedTime = 0.0;
    cudaEvent_t startDS, stopDS;
    cudaEventCreate(&startDS); CHECK_ERR;
    cudaEventCreate(&stopDS); CHECK_ERR;
    cudaEventRecord(startDS); CHECK_ERR;
    {f"{sparse_function_name}(A_sparse_dev, 0, B_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(nullptr, 0, B_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);"} CHECK_ERR;
    cudaEventRecord(stopDS); CHECK_ERR;
    cudaEventSynchronize(stopDS); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime, startDS, stopDS); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    std::cout << "Sparse x Dense kernel took " << elapsedTime << " ms" << std::endl; 
    cudaMemcpy(R2, C2_dev, sizeof(float)*{rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    std::cout << "Freeing device memory" << std::endl;
    cudaFree(B_dev);
    {f"cudaFree(A_sparse_dev);" if not with_compile_time_values else ""}
    cudaFree(A_dense_dev);
    cudaFree(C1_dev);
    cudaFree(C2_dev);

    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] >= R2[i] + 0.001 || R1[i] <= R2[i] - 0.001) {{
        //throw std::runtime_error("{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
        std::cout << "RESULTS DONT MATCH" << std::endl;
        return 0;
        }}
    }}
    std::cout << "{transA} Dense x {transB} Dense and {transA} Sparse x {transB} Dense Matrix Multiplications Match!" << std::endl;
    std::cout << "Results Match!" << std::endl;
    }}
    """
                    f = open(
                        f"{cur_dir}/cuda_code/benchmark_cuda_sparse_dense_{testid}.cu", "w")
                    f.write(s)
                    f.close()
                    # print(s)
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
