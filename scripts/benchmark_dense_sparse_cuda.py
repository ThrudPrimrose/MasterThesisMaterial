import random
from gemmforge import DenseMatrix, GenerationError, GemmGenerator, SparseMatrix
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType
from gemmforge.vm import vm_factory
import numpy as np
import sys
from random import randint
from numba import cuda
import os
import scipy

from params import *

# b_matrix_types = ["band", "single_column_b", "single_row_b", "chequered", "full"]
#"band", "single_column_b", "single_row_b", "chequered",
b_matrix_types = ["band", "single_column_b", "single_row_b", "chequered", "full", "random"]



def get_available_mem_on_gpu():
    gpus = cuda.gpus.lst

    # for gpu in gpus:
    gpu = gpus[0]
    meminfo = cuda.current_context().get_memory_info()
    # print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
    return meminfo[0]


def get_suggested_num_elements(MatASize, MatBDenseSize, MatBSparseSize, MatCSize, MatBCSCSize, SizeOfFloat):
    # We mul A x BD(dense) = C1, A x BS(Sparse) = C2
    # And compare C1 and C1, C1 and 2 obtained back will be R1 and R2 on host
    # On host we need A, BD, BS, C, R1, R2
    # On device we need A, BD, BS, C1, C2
    per_el_size = (MatASize + MatBDenseSize +
                   MatBSparseSize + MatCSize * 2 + MatBCSCSize) * SizeOfFloat

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    at95 = int(0.90 * can_fit_els)
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
    elif btype == "random":
        entry_count = int(non_zero_ratio * rowB * colB)
        b_el_count = entry_count
        l = set()
        while len(l) < entry_count:
            i = randint(0, rowB - 1)
            j = randint(0, colB - 1)
            l.add((i, j))
        llist = list(l)
        assert (len(llist) == b_el_count)
        for (row, col) in llist:
            B[row, col] = 1
        for j in range(colA):
            for i in range(rowA):
                if B[i, j] != 0:
                    r = random.randint(1, 9)
                    coo["coordinates"].append([i, j])
                    coo["entries"].append([i, j, r])
                    B[i, j] = r
        b_el_count = len(coo["coordinates"])
    else:
        raise Exception("NO")
    if btype != "random_entries":
        if transposed:
            npB = B
            B = B.flatten("C")
        else:
            npB = B
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
    return (coo, B, B_nonzeros, b_el_count, npB)


coo, B, B_nonzeros, b_el_count, npB = gen_matrix_b(9, 9, False, "band")
print(npB)
Bcsc = scipy.sparse.csc_matrix(npB, shape=(9,9), dtype=float, copy=True)
print(Bcsc.data)
print(", ".join([str(x) for x in Bcsc.data]))
print(", ".join([str(x) for x in Bcsc.indices]))
print(", ".join([str(x) for x in Bcsc.indptr]))

try:
    for with_compile_time_values in [True, False]:
        for b_type in b_matrix_types:
            for tA in [True, False]:
                for tB in [True, False]:
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
                    #if not tA:
                        rowC = row_c
                        colC = col_c
                    #else:
                    #    rowC = col_c
                    #    colC = row_c

                    mat_a = DenseMatrix(num_rows=rowA,
                                        num_cols=colA,
                                        addressing=adressingA,
                                        bbox=[0, 0, rowA, colA],
                                        )

                    coo, matrix_b, matrix_b_non_zeros_flat, b_el_count, npB = gen_matrix_b(
                        rowB, colB, tB, b_type)

                    mat_b_sparse = SparseMatrix(num_rows=rowB,
                                                num_cols=colB,
                                                addressing=adressingB,
                                                coordinates=coo["coordinates"],
                                                values=matrix_b_non_zeros_flat if with_compile_time_values else None)

                    mat_b_dense = DenseMatrix(num_rows=rowB,
                                              num_cols=colB,
                                              bbox=[0, 0, rowB, colB],
                                              addressing=adressingB)
                    Bcsc = scipy.sparse.csc_matrix(npB, shape=(rowB, colB), dtype=float, copy=True)
                    BCSC_data_str = "{" + ", ".join([str(x) for x in Bcsc.data]) + "}"
                    BCSC_indices_str = "{" +  ", ".join([str(x) for x in Bcsc.indices]) + "}"
                    BCSC_indptr_str = "{" + ", ".join([str(x) for x in Bcsc.indptr]) + "}"

                    mat_c = DenseMatrix(num_rows=rowC,
                                        num_cols=colC,
                                        bbox=[0, 0, rowC, colC],
                                        addressing=adressingC)

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
                    dense_gen.set(tA, tB, mat_a, mat_b_dense, mat_c, alpha=Alpha, beta=Beta,
                                  base_name=f"A{T if transA else NT}_B{T if transB else NT}_DenseXDense")
                    dense_gen.generate()
                    # print(dense_gen.get_kernel())
                    # print(dense_gen.get_launcher())
                    # print(dense_gen.get_launcher_header())
                    dense_header = dense_gen.get_launcher_header()
                    # Get the function name without void in the beginning
                    dense_function_name = dense_header.split("(")[0][4:]
                    dense_flops_per_op = dense_gen.get_flops()

                    # , kernel_type=GemmKernelType.DENSE_SPARSE_REGISTER_ONLY_FULL_UNIT_VECTOR_BASED
                    sparse_gen = GemmGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    sparse_gen.set(tA, tB, mat_a, mat_b_sparse, mat_c, alpha=Alpha, beta=Beta,
                                   base_name=f"A{T if transA else NT}_B{T if transB else NT}_{b_type}_DenseXSparse")
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
                    A = np.zeros(rowA * colA)
                    A.fill(1.0)
                    for i in range(rowA * colA):
                        A[i] = i * 2.0
                    B_dense = matrix_b
                    B_sparse = matrix_b_non_zeros_flat

                    np.set_printoptions(threshold=sys.maxsize)
                    strA = np.array2string(A, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strB_sparse = np.array2string(np.array(B_sparse), separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strB_dense = np.array2string(B_dense, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ').replace(
                        "[", "{").replace("]", "}")

                    get_available_mem_on_gpu()
                    MatBCSCSize = len(Bcsc.data) + len(Bcsc.indices) + len(Bcsc.indptr)

                    full, at95 = get_suggested_num_elements(
                        rowA * colA, rowB * colB, b_el_count, rowC * colC, MatBCSCSize, 4)
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
#include <cublas_v2.h>
#include <cusparse.h>

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

#define CHECK_CUSPARSE(func)                                                   \\
{{                                                                             \\
    cusparseStatus_t status = (func);                                          \\
    if (status != CUSPARSE_STATUS_SUCCESS) {{                                  \\
        printf("CUSPARSE API failed at line %d with error: %s (%d)\\n",         \\
               __LINE__, cusparseGetErrorString(status), status);              \\
        return EXIT_FAILURE;                                                   \\
    }}                                                                         \\
}}


// Dense x Dense Kernel
{dense_gen.get_kernel()}

// Dense x Sparse Kernel
{sparse_gen.get_kernel()}

// Dense x Dense Kernel Launcher
{dense_gen.get_launcher()}

// Dense x Sparse Kernel Launcher
{sparse_gen.get_launcher()}


__device__ const float alpha_dev = {Alpha}f;
__device__ const float beta_dev = {Beta}f;

int main(){{
    std::cout.precision(10);
    std::cout << std::setw(10);
    cublasHandle_t handle;
    cublasStatus_t createStatus = cublasCreate(&handle);
    if (createStatus != CUBLAS_STATUS_SUCCESS) {{
        throw std::runtime_error("UWU");
    }}
    int maxStreamCount = 25;
    if ({num_els} < maxStreamCount){{
        maxStreamCount = {num_els};
    }}
    cusparseHandle_t* cuSparseHandles = new cusparseHandle_t[maxStreamCount];
    cudaStream_t* cuSparseStreams = new cudaStream_t[maxStreamCount];

    std::cout << "Creatign streams" << std::endl;
    for (int i = 0; i < maxStreamCount; i++) {{
        cudaStreamCreate(&cuSparseStreams[i]);
        cusparseCreate(&cuSparseHandles[i]);
        cusparseSetStream(cuSparseHandles[i], cuSparseStreams[i]);
    }}

    float alpha = {Alpha}f;
    float beta = {Beta}f;

    std::cout << "Gemm-Type: " << "{n}" << std::endl;
    std::cout << "Number of elements: " << "{num_els}" << std::endl;
    std::cout << "Dense FLOP/s per element from gemmforge: " << "{dense_flops_per_op}" << std::endl;
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;
    float CoreA[{rowA}*{colA}] = {strA};
    float CoreB_sparse[{b_el_count}] = {strB_sparse};
    float CoreB_dense[{rowB} * {colB}] = {strB_dense};
    float CoreC[{rowC}*{colC}] = {strC};
    float CoreBCSC_data[{len(Bcsc.data)}] = {BCSC_data_str};
    float CoreBCSC_indices[{len(Bcsc.indices)}] = {BCSC_indices_str};
    float CoreBCSC_indptr[{len(Bcsc.indptr)}] = {BCSC_indptr_str};

    // Buffers 
    std::cout << "Instantiating buffer matrices" << std::endl;
    float* A = new float[{rowA}*{colA}*{num_els}];
    float* B_dense = new float[{rowB}*{colB}*{num_els}];
    {f"float* B_sparse = new float[{b_el_count}*{num_els}];" if not with_compile_time_values else ""}
    float* C = new float[{rowC}*{colC}*{num_els}];
    float* R1 = new float[{rowC}*{colC}*{num_els}];
    float* R2 = new float[{rowC}*{colC}*{num_els}];
    float* BCSC_data = new float[{len(Bcsc.data)}*{num_els}];
    int* BCSC_indices = new int[{len(Bcsc.indices)}*{num_els}];
    int* BCSC_indptr = new int[{len(Bcsc.indptr)}*{num_els}];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < {num_els}; i++){{
        std::memcpy(&A[{rowA} * {colA} * i], &CoreA[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&B_dense[{rowB} * {colB} * i], &CoreB_dense[0], {rowB} * {colB} * sizeof(float));
        {f"std::memcpy(&B_sparse[{b_el_count} * i], &CoreB_sparse[0], {b_el_count} * sizeof(float));" if not with_compile_time_values else ""}
        std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
        std::memcpy(&BCSC_data[{len(Bcsc.data)} * i], &CoreBCSC_data[0], {len(Bcsc.data)} * sizeof(int));
        std::memcpy(&BCSC_indices[{len(Bcsc.indices)} * i], &CoreBCSC_indices[0], {len(Bcsc.indices)} * sizeof(int));
        std::memcpy(&BCSC_indptr[{len(Bcsc.indptr)} * i], &CoreBCSC_indptr[0], {len(Bcsc.indptr)} * sizeof(int));
    }}

    float *A_dev = nullptr;
    {"float *B_sparse_dev = nullptr;" if not with_compile_time_values else ""}
    float *B_dense_dev = nullptr;
    float *C1_dev = nullptr;
    float *C2_dev = nullptr;
    int *BCSC_data_dev = nullptr;
    int *BCSC_indices_dev = nullptr;
    int *BCSC_indptr_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&A_dev, sizeof(float) * {rowA} * {colA} * {num_els}); CHECK_ERR;
    {f"cudaMalloc((void **)&B_sparse_dev, sizeof(float) * {b_el_count} * {num_els}); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMalloc((void **)&B_dense_dev, sizeof(float) * {rowB} * {colB} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&BCSC_data_dev, sizeof(int) * {len(Bcsc.data)} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&BCSC_indices_dev, sizeof(int) * {len(Bcsc.indices)} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&BCSC_indptr_dev, sizeof(int) * {len(Bcsc.indptr)} * {num_els}); CHECK_ERR;
    
    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {rowA} * {colA} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    {f"cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  {b_el_count} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  {rowB} * {colB} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)BCSC_data_dev, (void *)BCSC_data, sizeof(int) *  {len(Bcsc.data)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)BCSC_indices_dev, (void *)BCSC_indices, sizeof(int) * {len(Bcsc.indices)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)BCSC_indptr_dev, (void *)BCSC_indptr, sizeof(int) * {len(Bcsc.indptr)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    
    // Dense x Dense Matrix Mult
    {dense_function_name}(A_dev, 0, B_dense_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    {dense_function_name}(A_dev, 0, B_dense_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;

    std::cout << "Calling Dense x Dense kernel" << std::endl;
    float elapsedTime1 = 0.0; 
    cudaEvent_t startDD, stopDD;
    cudaEventCreate(&startDD); CHECK_ERR;
    cudaEventCreate(&stopDD); CHECK_ERR;
    cudaEventRecord(startDD); CHECK_ERR;
    //callFirstGemm(DeviceA, 0, DeviceB, 0, DeviceTmp, 0, NumElements, nullptr, FirstDriver.getTestStream());
    {dense_function_name}(A_dev, 0, B_dense_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr); CHECK_ERR;
    cudaEventRecord(stopDD); CHECK_ERR;
    cudaEventSynchronize(stopDD); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime1, startDD, stopDD); CHECK_ERR;
    std::cout << "Dense x Dense kernel took " << elapsedTime1 << " ms" << std::endl;
    cudaEventDestroy(startDD); CHECK_ERR;
    cudaEventDestroy(stopDD); CHECK_ERR;
    cudaMemcpy(R1, C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    // Dense x Sparse Matrix Mult
    {f"{sparse_function_name}(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(A_dev, 0, nullptr, 0, C2_dev, 0, {num_els}, nullptr, nullptr);"} CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    {f"{sparse_function_name}(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(A_dev, 0, nullptr, 0, C2_dev, 0, {num_els}, nullptr, nullptr);"} CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;

    std::cout << "Calling Dense x Sparse kernel" << std::endl;
    float elapsedTime2 = 0.0;
    cudaEvent_t startDS, stopDS;
    cudaEventCreate(&startDS); CHECK_ERR;
    cudaEventCreate(&stopDS); CHECK_ERR;
    cudaEventRecord(startDS); CHECK_ERR;
    {f"{sparse_function_name}(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(A_dev, 0, nullptr, 0, C2_dev, 0, {num_els}, nullptr, nullptr);"} CHECK_ERR;
    cudaEventRecord(stopDS); CHECK_ERR;
    cudaEventSynchronize(stopDS); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime2, startDS, stopDS); CHECK_ERR;
    std::cout << "Dense x Sparse kernel took " << elapsedTime2 << " ms" << std::endl; 
    cudaEventDestroy(startDS); CHECK_ERR;
    cudaEventDestroy(stopDS); CHECK_ERR;
    cudaMemcpy(R2, C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] >= R2[i] * 1.001 || R1[i] <= R2[i] * 0.999) {{
            std::string s = "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!";
            //throw std::runtime_error(s);
            std::cout << "RESULTS DONT MATCH" << std::endl;
            std::cout << s << std::endl;
            return 0;
        }}
    }}
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;

    std::cout << "Calling cuBlas DxD Kernels" << std::endl;
    float elapsedTime3 = 0.0;
    float** A_begins = new float*[{num_els}];
    float** B_dense_begins = new float*[{num_els}];
    float** C2_begins = new float*[{num_els}];
    float** A_dev_begins = nullptr;
    float** B_dense_dev_begins = nullptr;
    float** C2_dev_begins = nullptr;
    for (int i = 0; i < {num_els}; i++){{
        A_begins[i]  = &A_dev[0]  + {rowA} * {colA} * i;
        B_dense_begins[i]  = &B_dense_dev[0]  + {rowB} * {colB} * i;
        C2_begins[i] = &C2_dev[0] + {rowC} * {colC} * i;
    }}
    cudaMalloc((void **)&A_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMalloc((void **)&B_dense_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMalloc((void **)&C2_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMemcpy((void *)A_dev_begins, (void *)A_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)B_dense_dev_begins, (void *)B_dense_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C2_dev_begins, (void *)C2_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    // First 2 to discard
    cublasStatus_t cuBlasStatus = cublasSgemmBatched(handle, {"CUBLAS_OP_T" if transA else "CUBLAS_OP_N"}, {"CUBLAS_OP_T" if transB else "CUBLAS_OP_N"},
        {rowA}, {colB}, {colA}, &alpha_dev, (const float **)A_dev_begins, {rowA}, (const float **)B_dense_dev_begins, {rowB},
        &beta_dev, (float **)C2_dev_begins, {rowC}, {num_els}); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    if (cuBlasStatus != CUBLAS_STATUS_SUCCESS) {{
        std::cout << "First cuBlas call failed";
    }}
    cuBlasStatus = cublasSgemmBatched(handle, {"CUBLAS_OP_T" if transA else "CUBLAS_OP_N"}, {"CUBLAS_OP_T" if transB else "CUBLAS_OP_N"},
            {rowA}, {colB}, {colA}, &alpha_dev, (const float **)A_dev_begins, {rowA}, (const float **)B_dense_dev_begins, {rowB},
            &beta_dev, (float **)C2_dev_begins, {rowC}, {num_els}); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    if (cuBlasStatus != CUBLAS_STATUS_SUCCESS) {{
        std::cout << "Second cuBlas call failed";
    }}
    cudaEvent_t startCuBlas, stopCuBlas; 
    cudaEventCreate(&startCuBlas); CHECK_ERR;
    cudaEventCreate(&stopCuBlas); CHECK_ERR;
    cudaEventRecord(startCuBlas); CHECK_ERR;
    cuBlasStatus = cublasSgemmBatched(handle, {"CUBLAS_OP_T" if transA else "CUBLAS_OP_N"}, {"CUBLAS_OP_T" if transB else "CUBLAS_OP_N"},
        {rowA}, {colB}, {colA}, (const float*)&alpha_dev, (const float **)A_dev_begins, {rowA}, (const float **)B_dense_dev_begins, {rowB},
        (const float*)&beta_dev, (float **)C2_dev_begins, {rowC}, {num_els}); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    if (cuBlasStatus != CUBLAS_STATUS_SUCCESS) {{
        std::cout << "Second cuBlas call failed";
    }}
    cudaEventRecord(stopCuBlas); CHECK_ERR;
    cudaEventSynchronize(stopCuBlas); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime3, startCuBlas, stopCuBlas); CHECK_ERR;
    std::cout << "cuBlas DxD kernel took " << elapsedTime3 << " ms" << std::endl; 
    cudaEventDestroy(startCuBlas); CHECK_ERR;
    cudaEventDestroy(stopCuBlas); CHECK_ERR;
    cudaMemcpy(R2, C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] >= R2[i] * 1.001 || R1[i] <= R2[i] * 0.999) {{
            std::string s = "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + " ->" + 
                std::to_string(R1[i]) + " and " + std::to_string(R2[i]) + " | " + std::to_string(R1[{rowC}*{colC}*{num_els}-1]) + " and " + std::to_string(R2[{rowC}*{colC}*{num_els} - 1]);
            //throw std::runtime_error(s);
            std::cout << "RESULTS DONT MATCH" << std::endl;
            std::cout << s << std::endl;
            return 0;
        }}
    }}

    delete[] A_begins;
    delete[] B_dense_begins;
    delete[] C2_begins;
    cudaFree(A_dev_begins); CHECK_ERR;
    cudaFree(C2_dev_begins); CHECK_ERR;
    cudaFree(B_dense_dev_begins); CHECK_ERR;

    std::cout << "Calling cuSparse DxS Kernels" << std::endl;
    float elapsedTime4 = 0.0;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    cusparseDnMatDescr_t matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, {rowB}, {colB}, {len(Bcsc.data)}, BCSC_indices_dev, 
        BCSC_indptr_dev, BCSC_data_dev, 
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, {colA}, {rowA}, {colA}, A_dev , CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, {colC}, {rowC}, {colC}, C2_dev , CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matA, {num_els}, {rowA}*{colA}));
    CHECK_CUSPARSE( cusparseCsrSetStridedBatch(matB, {num_els}, 0, {len(Bcsc.data)}));
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matC, {num_els}, {rowC}*{colC}));
    cusparseOperation_t cuSparseTransA = {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not transA else "CUSPARSE_OPERATION_TRANSPOSE"};
    cusparseOperation_t cuSparseTransB = {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not transB else "CUSPARSE_OPERATION_TRANSPOSE"};
    cudaEvent_t startCuSparse, stopCuSparse;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 cuSparseHandles[0],
                                 cuSparseTransA,
                                 cuSparseTransB,
                                 &alpha, matB, matA, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, &bufferSize) );
    cudaMalloc(&dBuffer, bufferSize); CHECK_ERR;
    cudaEventCreate(&startCuSparse); CHECK_ERR;
    cudaEventCreate(&stopCuSparse); CHECK_ERR;
    cudaEventRecord(startCuSparse); CHECK_ERR;
    CHECK_CUSPARSE( cusparseSpMM(cuSparseHandles[0], cuSparseTransA, cuSparseTransB, 
        &alpha, matB, matA, &beta, matC, 
        CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, dBuffer));
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaEventRecord(stopCuSparse); CHECK_ERR;
    cudaEventSynchronize(stopCuSparse); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTime4, startCuSparse, stopCuSparse); CHECK_ERR;
    std::cout << " cuSparse DxD kernel took " << elapsedTime4 << " ms" << std::endl; 
    cudaEventDestroy(startCuSparse); CHECK_ERR;
    cudaEventDestroy(stopCuSparse); CHECK_ERR;
    cudaMemcpy(R2, C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;
    for (int i = 0; i < maxStreamCount; i++) {{
        cusparseDestroy(cuSparseHandles[i]); CHECK_ERR;
        cudaStreamDestroy(cuSparseStreams[i]); CHECK_ERR;
    }}

    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] >= R2[i] * 1.001 || R1[i] <= R2[i] * 0.999) {{
        //throw std::runtime_error("{transA} Dense x {transB} Dense and CUSPARSE {transA} Dense x {transB} Dense Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
        std::cout << "RESULTS DONT MATCH" << std::endl;
        std::cout << "For these benchmarks, a transpose is needed" << std::endl;
        break;
        }}
    }}

    std::cout << "Freeing device memory" << std::endl;
    cudaFree(A_dev);
    {f"cudaFree(B_sparse_dev);" if not with_compile_time_values else ""}
    cudaFree(B_dense_dev);
    cudaFree(C1_dev);
    cudaFree(C2_dev);

    delete[] R1;
    delete[] R2;
    delete[] A;
    delete[] B_dense;
    delete[] C;
    {f"delete[] B_sparse;" if not with_compile_time_values else ""}

    //std::cout << "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Multiplications Match!" << std::endl;
    //std::cout << "Results Match!" << std::endl;
    }}
    """
                    f = open(
                        f"{scripts_dir}/cuda_code/benchmark_cuda_dense_sparse_{testid}.cu", "w")
                    f.write(s)
                    f.close()
                    # print(s)
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
