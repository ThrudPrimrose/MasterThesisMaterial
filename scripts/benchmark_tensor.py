from gemmforge import DenseMatrix, GenerationError, GemmGenerator, LoGGenerator
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType
from gemmforge.vm import vm_factory
import numpy as np
import sys
from random import randint
from numba import cuda
import os

from params import *

# b_matrix_types = ["band", "single_column_b", "single_row_b", "chequered", "full"]
b_matrix_types = ["band", "single_column_b",
                  "single_row_b", "chequered", "full"]

def get_available_mem_on_gpu():
    gpus = cuda.gpus.lst

    # for gpu in gpus:
    gpu = gpus[0]
    meminfo = cuda.current_context().get_memory_info()
    # print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
    return meminfo[0]


def get_suggested_num_elements(MatASize, MatBDenseSize, MatCSize, SizeOfFloat):
    # We mul A x BD(dense) = C1, A x BS(Sparse) = C2
    # And compare C1 and C1, C1 and 2 obtained back will be R1 and R2 on host
    # On host we need A, BD, BS, C, R1, R2
    # On device we need A, BD, BS, C1, C2
    per_el_size = (MatASize + MatBDenseSize + MatCSize * 2) * SizeOfFloat

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    at95 = int(0.95 * can_fit_els)
    # print(f"Can fit {can_fit_els} matrices of given sizes, at 80% capacity {at80}")
    return (can_fit_els, at95)
    # return (1,1)


def gen_matrix_b(rowB, colB, transposed, btype):
    B = np.zeros([rowB, colB])
    coo = {"name": "B", "rows": rowB, "cols": colB,
           "entries": [], "coordinates": []}

    if btype == "band":
        if not transposed:
            coo["entries"].append([0, 0, 2.0])
            coo["coordinates"].append([0, 0])
            coo["entries"].append([1, 0, 1.0])
            coo["coordinates"].append([1, 0])
            for i in range(1, rowB - 1):
                coo["entries"].append([i - 1, i, 3.0])
                coo["coordinates"].append([i - 1, i])
                coo["entries"].append([i, i, 2.0])
                coo["coordinates"].append([i, i])
                coo["entries"].append([i + 1, i, 1.0])
                coo["coordinates"].append([i + 1, i])
            i = rowB - 1
            coo["entries"].append([i - 1, i, 3.0])
            coo["coordinates"].append([i - 1, i])
            coo["entries"].append([i, i, 2.0])
            coo["coordinates"].append([i, i])
        else:
            coo["entries"].append([0, 0, 2.0])
            coo["coordinates"].append([0, 0])
            coo["entries"].append([0, 1, 3.0])
            coo["coordinates"].append([0, 1])
            for i in range(1, rowB - 1):
                coo["entries"].append([i, i - 1, 1.0])
                coo["coordinates"].append([i, i - 1])
                coo["entries"].append([i, i, 2.0])
                coo["coordinates"].append([i, i])
                coo["entries"].append([i, i + 1, 3.0])
                coo["coordinates"].append([i, i + 1])
            i = rowB - 1
            coo["entries"].append([i, i - 1, 1.0])
            coo["coordinates"].append([i, i - 1])
            coo["entries"].append([i, i, 2.0])
            coo["coordinates"].append([i, i])

        for i in range(rowB):
            B[i, i] = 2.0
        for i in range(rowB - 1):
            B[i, i + 1] = 3.0
        for i in range(1, rowB):
            B[i, i - 1] = 1.0
        b_el_count = 2 * 2 + 3 * (rowB - 2)
    elif btype == "single_column_b":
        at = 1
        for i in range(rowB):
            B[i, at] = i + 1.0
        for i in range(rowB):
            coo["entries"].append([i, at, i + 1.0])
            coo["coordinates"].append([i, at])
        b_el_count = rowB
    elif btype == "single_row_b":
        at = 1
        for j in range(colB):
            B[at, j] = j + 1.0
        for j in range(colB):
            coo["entries"].append([at, j, j + 1.0])
            coo["coordinates"].append([at, j])
        b_el_count = colB
    elif btype == "chequered":
        npB = np.zeros((rowB, colB))
        if transposed:
            for i in range(rowB):
                offset = i % 2
                for j in range(offset, colB, 2):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i * 10.0 + j + 1
                    B[i, j] = i * 10.0 + j + 1
        else:
            for j in range(colB):
                offset = j % 2
                for i in range(offset, rowB, 2):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i * 10.0 + j + 1
                    B[i, j] = i * 10.0 + j + 1
        b_el_count = len(coo["coordinates"])
    elif btype == "box":
        npB = np.zeros((rowB, colB))
        if transposed:
            for i in range(int(rowB / 2)):
                for j in range(int(colB / 2)):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i * 10.0 + j + 1
                    B[i, j] = i * 10.0 + j + 1
        else:
            for j in range(int(colB / 2)):
                for i in range(int(rowB / 2)):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i * 10.0 + j + 1
                    B[i, j] = i * 10.0 + j + 1
        b_el_count = len(coo["coordinates"])
    elif btype == "full":
        if transposed:
            for i in range(colB):
                for j in range(rowB):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    B[i, j] = i * 10.0 + j + 1
        else:
            for j in range(colB):
                for i in range(rowB):
                    coo["entries"].append([i, j, i * 10.0 + j + 1])
                    coo["coordinates"].append([i, j])
                    B[i, j] = i * 10.0 + j + 1
        b_el_count = len(coo["coordinates"])
    else:
        raise Exception("NO")
    if btype != "random_entries":
        if transposed:
            Bo = B
            B = B.flatten("C")
        else:
            Bo = B
            B = B.flatten("F")
        T = "T"
        NT = ""
        # print(btype, f"{T if transposed else NT}: ", coo["coordinates"])
        # print(btype, f"{T if transposed else NT}: ", Bo)
        B_nonzeros = []
        for el in B:
            if el != 0.0:
                B_nonzeros.append(el)
        # print(btype, f"{T if transposed else NT} sparse: ", B_nonzeros)
    else:
        B_nonzeros = []
    return (coo, B, B_nonzeros, b_el_count)


try:
    for with_compile_time_values in [True, False]:
        for b_type in b_matrix_types:
            for tA in [True]:
                for tB in [False]:
                    testid = ""
                    if tA:
                        testid += "At_mul_"
                    else:
                        testid += "A_mul_"
                    if tB:
                        testid += "Bt"
                    else:
                        testid += "B"
                    testid += "_" + b_type
                    valid = "_compiler_time_value" if with_compile_time_values else ""
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

                    # else:
                    # rowC = 9
                    # colC = 56
                    # rowA = 64
                    # colA = 32
                    # rowB = 32
                    # colB = 32
                    # rowC = 64
                    # colC = 32

                    mat_a = DenseMatrix(num_rows=rowA,
                                        num_cols=colA,
                                        addressing=adressingA,
                                        bbox=[0, 0, rowA, colA],
                                        leading_dimension=rowA)

                    coo, matrix_b, matrix_b_non_zeros_flat, b_el_count = gen_matrix_b(
                        rowB, colB, tB, b_type)

                    mat_b = DenseMatrix(num_rows=rowB,
                                        num_cols=colB,
                                        bbox=[0, 0, rowB, colB],
                                        addressing=adressingB,
                                        leading_dimension=rowB)

                    mat_c = DenseMatrix(num_rows=rowC,
                                        num_cols=colC,
                                        bbox=[0, 0, rowC, colC],
                                        addressing=adressingC,
                                        leading_dimension=rowC)

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
                    dense_gen = LoGGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    dense_gen.set(tA, tB, mat_a, mat_b, mat_c, alpha=Alpha, beta=Beta,
                                  base_name=f"A{T if transA else NT}_B{T if transB else NT}_DenseXDense")
                    dense_gen.generate()
                    # print(dense_gen.get_kernel())
                    # print(dense_gen.get_launcher())
                    # print(dense_gen.get_launcher_header())
                    dense_header = dense_gen.get_launcher_header()
                    # Get the function name without void in the beginning
                    dense_function_name = dense_header.split("(")[0][4:]
                    dense_flops_per_op = dense_gen.get_flops()

                    # A = np.random.random({rowA} * 9)
                    # B = np.random.random(9 * 9)
                    C = np.zeros(rowC * colC)
                    C.fill(0.1)
                    for i in range(rowC * colC):
                        C[i] = i * 0.1
                    A = np.zeros(rowA * colA)
                    A.fill(1.0)
                    for i in range(rowA * colA):
                        A[i] = i * 2.0
                    B = matrix_b

                    np.set_printoptions(threshold=sys.maxsize)
                    strA = np.array2string(A, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strB = np.array2string(B, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ').replace(
                        "[", "{").replace("]", "}")

                    get_available_mem_on_gpu()
                    full, at95 = get_suggested_num_elements(
                        rowA * colA * 32, rowB * colB * 32, rowC * colC * 32, 4)
                    num_els = at95
                    # num_els = 1
                    # num_els = 104857

                    ctv = "_ctv" if with_compile_time_values else ""
                    n = f"A{T if transA else NT}_B{T if transB else NT}_{b_type}_DenseXSparse{ctv}"
                    # print(n)
                    s = f"""
    #include <iostream>
    #include <cuda_runtime.h>
    #include <cstring>
    #include <iomanip>

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

    // Dense x Dense Kernel Launcher
    {dense_gen.get_launcher()}

    int main(){{
    std::cout.precision(10);
    std::cout << std::setw(10);

    std::cout << "Gemm-Type: " << "{n}" << std::endl;
    std::cout << "Number of elements: " << "{num_els}" << std::endl;
    std::cout << "Dense FLOP/s per element from gemmforge: " << "{dense_flops_per_op}" << std::endl;
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;
    float CoreA[{rowA}*{colA}] = {strA};
    float CoreB[{rowB}*{colB}] = {strB};
    float CoreC[{rowC}*{colC}] = {strC};
    
    // Buffers 
    std::cout << "Instantiating buffer matrices" << std::endl;
    float* A = new float[{rowA}*{colA}*32*{num_els}];
    float* B = new float[{rowB}*{colB}*32*{num_els}];
    float* C = new float[{rowC}*{colC}*32*{num_els}];
    float* R1 = new float[{rowC}*{colC}*32*{num_els}];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < 32*{num_els}; i++){{
        std::memcpy(&A[{rowA} * {colA} * i], &CoreA[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&B[{rowB} * {colB} * i], &CoreB[0], {rowB} * {colB} * sizeof(float));
        std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
    }}

    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C1_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&A_dev, sizeof(float) * {rowA} * {colA} *32* {num_els}); CHECK_ERR;
    cudaMalloc((void **)&B_dev, sizeof(float) * {rowB} * {colB} *32* {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC} *32* {num_els}); CHECK_ERR;

    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {rowA} * {colA} *32* {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) *  {rowB} * {colB} *32* {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC} *32* {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;

    // Dense x Dense Matrix Mult
    std::cout << "Fire and forget kernels for warm-up" << std::endl;
    {dense_function_name}(A_dev, 0, B_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr);
    cudaDeviceSynchronize();
    {dense_function_name}(A_dev, 0, B_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr);
    cudaDeviceSynchronize();

    std::cout << "Calling Dense x Dense kernel" << std::endl;
    float elapsedTime1 = 0.0; 
    cudaEvent_t startDD, stopDD;
    cudaEventCreate(&startDD);
    cudaEventCreate(&stopDD);
    cudaEventRecord(startDD);
    //callFirstGemm(DeviceA, 0, DeviceB, 0, DeviceTmp, 0, NumElements, nullptr, FirstDriver.getTestStream());
    {dense_function_name}(A_dev, 0, B_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr);
    cudaEventRecord(stopDD);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stopDD);
    cudaEventElapsedTime(&elapsedTime1, startDD, stopDD);
    std::cout << "Dense x Dense kernel took " << elapsedTime1 << " ms" << std::endl;
    cudaEventDestroy(startDD);
    cudaEventDestroy(stopDD);
    cudaMemcpy(R1, C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost);

    std::cout << "Freeing device memory" << std::endl;
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C1_dev);

    delete[] R1;
    delete[] A;
    delete[] B;
    delete[] C;

    }}
    """
                    f = open(
                        f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_{testid}.cu", "w")
                    f.write(s)
                    f.close()
                    # print(s)
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
