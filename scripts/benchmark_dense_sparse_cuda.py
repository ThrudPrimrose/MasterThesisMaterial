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
                   MatBSparseSize + MatCSize * 5 + MatBCSCSize) * SizeOfFloat
    # 3 extra Matrix C because of cuSparse...

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
        if not transposed:
            for j in range(colA):
                for i in range(rowA):
                    if B[i, j] != 0:
                        r = random.randint(1, 9)
                        coo["coordinates"].append([i, j])
                        coo["entries"].append([i, j, r])
                        B[i, j] = r
        else:
            for i in range(rowA):
                for j in range(colA):
                    if B[i, j] != 0:
                        r = random.randint(1, 9)
                        coo["coordinates"].append([i, j])
                        coo["entries"].append([i, j, r])
                        B[i, j] = r
        b_el_count = len(coo["coordinates"])
    else:
        raise Exception("NO")


    if btype != "random":
        npB = B
        B = B.flatten("F")
        B_nonzeros = []
        for el in B:
            if el > 0.0001 or el < -0.0001:
                assert (el != 0 and el != 0.0)
                B_nonzeros.append(el)
    else:
        if transposed:
            npB = B
            B = npB.flatten("C")
            B_nonzeros = []
            for el in B:
                if el > 0.0001 or el < -0.0001:
                    assert (el != 0 and el != 0.0)
                    B_nonzeros.append(el)
        else:
            npB = B
            B = B.flatten("F")
            B_nonzeros = []
            for el in B:
                if el > 0.0001 or el < -0.0001:
                    assert (el != 0 and el != 0.0)
                    B_nonzeros.append(el)
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
                        rowAT = col_a
                        colAT = row_a
                    else:
                        rowA = col_a
                        colA = row_a
                        rowAT = row_a
                        colAT = col_a
                    if not tB:
                        rowB = row_b
                        colB = col_b
                        rowBT = col_b
                        colBT = row_b
                    else:
                        rowB = col_b
                        colB = row_b
                        rowBT = col_b
                        colBT = row_b
                    #if not tA:
                    rowC = row_c
                    colC = col_c
                    rowCT = col_c
                    colCT = row_c
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
                    if tB:
                        BT = npB.transpose()
                    else:
                        BT = npB
                    BT_1d = BT.flatten("F")
                    BT_tmp = []
                    for b in BT_1d:
                        if b != 0.0:
                           BT_tmp.append(b)
                    BT_1d = np.array(BT_tmp)

                    BT_sparse = scipy.sparse.csr_matrix(BT)
                    BT_sparse_data = BT_sparse.data
                    BT_sparse_indices = BT_sparse.indices
                    BT_sparse_indptr = BT_sparse.indptr

                    BT_sparse_data_str = "{" + ", ".join([str(x) for x in BT_sparse_data]) + "}"
                    BT_sparse_indices_str = "{" +  ", ".join([str(x) for x in BT_sparse_indices]) + "}"
                    BT_sparse_indptr_str = "{" + ", ".join([str(x) for x in BT_sparse_indptr]) + "}"
                    print("data:", BT_sparse_data_str)
                    print("indices:", BT_sparse_indices_str)
                    print("indptr:", BT_sparse_indptr_str)

                    CTId_sparse = scipy.sparse.csr_matrix(np.zeros((colC, colC)))
                    CTId_sparse_data = CTId_sparse.data
                    CTId_sparse_indices = CTId_sparse.indices
                    CTId_sparse_indptr = CTId_sparse.indptr
                    CTId_sparse_data_str = "{" + ", ".join([str(x) for x in CTId_sparse_data]) + "}"
                    CTId_sparse_indices_str = "{" +  ", ".join([str(x) for x in CTId_sparse_indices]) + "}"
                    CTId_sparse_indptr_str = "{" + ", ".join([str(x) for x in CTId_sparse_indptr]) + "}"

                    mat_b_sparse = SparseMatrix(num_rows=rowB,
                                                num_cols=colB,
                                                addressing=adressingB,
                                                coordinates=coo["coordinates"],
                                                values=matrix_b_non_zeros_flat if with_compile_time_values else None)

                    mat_b_dense = DenseMatrix(num_rows=rowB,
                                              num_cols=colB,
                                              bbox=[0, 0, rowB, colB],
                                              addressing=adressingB)

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
                    if tA:
                        AT = A.transpose()
                    else:
                        AT = A
                    AT_1d = AT.flatten("F")
                    CT = C.transpose()
                    CT_1d = CT.flatten("F")
                    B_dense = matrix_b
                    B_sparse = matrix_b_non_zeros_flat

                    Bid = np.identity(colC)
                    Bidstr = np.array2string(Bid, separator=', ').replace(
                        "[", "{").replace("]", "}")

                    np.set_printoptions(threshold=sys.maxsize)
                    strA = np.array2string(A, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strB_sparse = np.array2string(np.array(B_sparse), separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strB_dense = np.array2string(B_dense, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strBT = np.array2string(BT_1d, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strAT = np.array2string(AT_1d, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    strCT = np.array2string(CT_1d, separator=', ').replace(
                        "[", "{").replace("]", "}")
                    get_available_mem_on_gpu()
                    MatBCSCSize = len(BT_sparse_data) + len(BT_sparse_indices) + len(BT_sparse_indptr)

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

void transpose_matrix(int M, int K, float* A, float* AT){{
    for (int i = 0; i < M; ++i) {{
        for (int j = 0; j < K; ++j) {{
            AT[i*K + j] = A[j*M + i];
        }}
    }}
}}


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
                                          

#define CHECK_CUDA(func)                                                       \\
{{                                                                       \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {{                                            \\
        printf("CUDA API failed at line %d with error: %s (%d)\\n",             \\
               __LINE__, cudaGetErrorString(status), status);                  \\
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


const float alpha_dev = {Alpha}f;
const float beta_dev = {Beta}f;
const float transpose_alpha_dev = 1.0f;
const float transpose_beta_dev = 0.0f;

int main(){{
    std::cout.precision(10);
    std::cout << std::setw(10);
    cublasHandle_t handle;
    cublasStatus_t createStatus = cublasCreate(&handle);
    if (createStatus != CUBLAS_STATUS_SUCCESS) {{
        throw std::runtime_error("UWU");
    }}
    cusparseHandle_t cuSparseHandle;
    cusparseCreate(&cuSparseHandle);

    std::cout << "Gemm-Type: " << "{n}" << std::endl;
    std::cout << "Number of elements: " << "{num_els}" << std::endl;
    std::cout << "Dense FLOP/s per element from gemmforge: " << "{dense_flops_per_op}" << std::endl;
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;

    float CoreA[{rowA} * {colA}] = {strA};
    float CoreB_sparse[{b_el_count}] = {strB_sparse};
    float CoreB_dense[{rowB} * {colB}] = {strB_dense};
    float CoreC[{rowC} * {colC}] = {strC};
    float CoreBT[{rowB} * {colB}] = {strBT};
    float CoreAT[{rowA} * {colA}] = {strAT};
    float CoreCT[{rowC} * {colC}] = {strCT};

    float CoreBT_data[{len(BT_sparse_data)}] = {BT_sparse_data_str};
    float CoreBT_indices[{len(BT_sparse_indices)}] = {BT_sparse_indices_str};
    float CoreBT_indptr[{len(BT_sparse_indptr)}] = {BT_sparse_indptr_str};

    // Buffers 
    std::cout << "Instantiating buffer matrices" << std::endl;
    float* A = new float[{rowA}*{colA}*{num_els}];
    float* B_dense = new float[{rowB}*{colB}*{num_els}];
    {f"float* B_sparse = new float[{b_el_count}*{num_els}];" if not with_compile_time_values else ""}
    float* C = new float[{rowC}*{colC}*{num_els}];
    float* R1 = new float[{rowC}*{colC}*{num_els}];
    float* R2 = new float[{rowC}*{colC}*{num_els}];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < {num_els}; i++){{
        std::memcpy(&A[{rowA} * {colA} * i], &CoreA[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&B_dense[{rowB} * {colB} * i], &CoreB_dense[0], {rowB} * {colB} * sizeof(float));
        {f"std::memcpy(&B_sparse[{b_el_count} * i], &CoreB_sparse[0], {b_el_count} * sizeof(float));" if not with_compile_time_values else ""}
        std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
    }}

    float *A_dev = nullptr;
    {"float *B_sparse_dev = nullptr;" if not with_compile_time_values else ""}
    float *B_dense_dev = nullptr;
    float *C1_dev = nullptr;
    float *C2_dev = nullptr;
    //int *BCSC_data_dev = nullptr;
    //int *BCSC_indices_dev = nullptr;
    //int *BCSC_indptr_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&A_dev, sizeof(float) * {rowA} * {colA} * {num_els}); CHECK_ERR;
    {f"cudaMalloc((void **)&B_sparse_dev, sizeof(float) * {b_el_count} * {num_els}); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMalloc((void **)&B_dense_dev, sizeof(float) * {rowB} * {colB} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    
    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {rowA} * {colA} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    {f"cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  {b_el_count} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  {rowB} * {colB} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;

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
    delete[] A;
    delete[] B_dense;
    delete[] R2;
    cudaFree(C2_dev);

    // Buffers 
    std::cout << "Instantiating transposed buffer matrices" << std::endl;
    float* AT = new float[{rowA}*{colA}*{num_els}];
    float* CT = new float[{rowC}*{colC}*{num_els}];
    float* RT = new float[{rowC}*{colC}*{num_els}];
    float* BT_data = new float[{len(BT_sparse_data)}*{num_els}];
    int* BT_indices = new int[{len(BT_sparse_indices)}*{num_els}];
    int* BT_indptr = new int[{len(BT_sparse_indptr)}*{num_els}];
    float* IdMat = new float[{colC}*{colC}*{num_els}];

    for (int k = 0; k < {num_els}; k++){{
        for (int i = 0; i < {colC}; i++){{
            for (int j = 0; j < {colC}; j++){{
                if (i == j){{
                    IdMat[k*{colC}*{colC} + i*{colC} + j ] = 1.0f;
                }}
                if (i!=j){{
                    IdMat[k*{colC}*{colC} + i*{colC} + j ] = 0.0f;
                }}
                
            }}
        }}
    }}


    std::cout << "Copying transposed core matrices to buffers" << std::endl;
    for (int i = 0; i < {num_els}; i++){{
        std::memcpy(&AT[{rowA} * {colA} * i], &CoreAT[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&CT[{rowC} * {colC} * i], &CoreCT[0], {rowC} * {colC} * sizeof(float));
        std::memcpy(&BT_data[{len(BT_sparse_data)} * i], &CoreBT_data[0], {len(BT_sparse_data)} * sizeof(int));
        std::memcpy(&BT_indices[{len(BT_sparse_indices)} * i], &CoreBT_indices[0], {len(BT_sparse_indices)} * sizeof(int));
        std::memcpy(&BT_indptr[{len(BT_sparse_indptr)} * i], &CoreBT_indptr[0], {len(BT_sparse_indptr)} * sizeof(int));
    }}

    std::cout << "Calling cuSparse DxS Kernels" << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cusparseSetStream(cuSparseHandle, stream);
    cublasSetStream(handle, stream);

    cusparseSpMatDescr_t matBT;
    cusparseDnMatDescr_t matAT, matCT;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float* BT_data_dev = nullptr;
    int* BT_indices_dev = nullptr;
    int* BT_indptr_dev = nullptr;
    float* C3_dev = nullptr;
    float* C4_dev = nullptr;
    float* IdMat_dev = nullptr;
    cudaMalloc((void **)&BT_data_dev, {num_els} * {len(BT_sparse_data)} * sizeof(float)); CHECK_ERR;
    cudaMalloc((void **)&BT_indices_dev, {num_els} * {len(BT_sparse_indices)} * sizeof(int)); CHECK_ERR;
    cudaMalloc((void **)&BT_indptr_dev, {num_els} * {len(BT_sparse_indptr)} * sizeof(int)); CHECK_ERR;
    cudaMalloc((void **)&C3_dev, {num_els} * {rowC} * {colC} * sizeof(float)); CHECK_ERR;
    cudaMalloc((void **)&C4_dev, {num_els} * {rowC} * {colC} * sizeof(float)); CHECK_ERR;
    cudaMalloc((void **)&IdMat_dev, {num_els} * {rowC} * {colC} * sizeof(float)); CHECK_ERR;
    cudaMemcpy((void *)BT_data_dev, (void *)BT_data, sizeof(float) *  {len(BT_sparse_data)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)BT_indices_dev, (void *)BT_indices, sizeof(int) * {len(BT_sparse_indices)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)BT_indptr_dev, (void *)BT_indptr, sizeof(int) * {len(BT_sparse_indptr)} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C3_dev, (void *)C, sizeof(float) * {rowCT} * {colCT} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)IdMat_dev, (void *)IdMat, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    float** C3_begins = new float*[{num_els}];
    float** C3_dev_begins = nullptr;
    float** C4_begins = new float*[{num_els}];
    float** C4_dev_begins = nullptr;
    float** IdMat_begins = new float*[{num_els}];
    float** IdMat_dev_begins = nullptr;
    for (int i = 0; i < {num_els}; i++){{
        C3_begins[i] = &C3_dev[0] + {rowC} * {colC} * i;
        C4_begins[i] = &C4_dev[0] + {rowC} * {colC} * i;
        IdMat_begins[i] = &IdMat_dev[0] + {rowC} * {colC} * i;
    }}
    cudaMalloc((void **)&C3_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMalloc((void **)&C4_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMalloc((void **)&IdMat_dev_begins, {num_els} * sizeof(float*)); CHECK_ERR;
    cudaMemcpy((void *)C3_dev_begins, (void *)C3_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C4_dev_begins, (void *)C4_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)IdMat_dev_begins, (void *)IdMat_begins, {num_els} * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
    CHECK_CUSPARSE( cusparseCreateCsr(&matBT, {colBT}, {rowBT}, {len(BT_sparse_data)},
                                      BT_indptr_dev, BT_indices_dev, BT_data_dev,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCsrSetStridedBatch(matBT, {num_els}, {len(BT_sparse_indptr)}, {len(BT_sparse_data)}))

    CHECK_CUSPARSE( cusparseCreateDnMat(&matAT, {rowA}, {colA}, {rowA}, A_dev,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matAT,  {num_els}, {rowA} * {colA}) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matCT, {colCT}, {rowCT}, {colCT}, C3_dev,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matCT, {num_els}, {rowCT} * {colCT}) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 cuSparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 { "CUSPARSE_OPERATION_NON_TRANSPOSE" if tA else "CUSPARSE_OPERATION_TRANSPOSE"},
                                 (const float*)&alpha_dev, matBT, matAT, (const float*)&beta_dev, matCT, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    //cudaDeviceSynchronize(); CHECK_ERR;
    // execute SpMM
    cudaEvent_t startcuSparse, stopcuSparse;
    cudaEventCreate(&startcuSparse); CHECK_ERR;
    cudaEventCreate(&stopcuSparse); CHECK_ERR;
    cudaEventRecord(startcuSparse); CHECK_ERR;
    CHECK_CUSPARSE( cusparseSpMM(cuSparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 { "CUSPARSE_OPERATION_NON_TRANSPOSE" if tA else "CUSPARSE_OPERATION_TRANSPOSE"},
                                 &alpha_dev, matBT, matAT, &beta_dev, matCT, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, dBuffer) )
    //cudaDeviceSynchronize(); CHECK_ERR;
    cuBlasStatus = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        {rowC}, {colC}, {rowC}, (const float*)&transpose_alpha_dev, (const float **)C3_dev_begins, {rowC}, (const float **)IdMat_dev_begins, {rowC},
        (const float*)&transpose_beta_dev, (float **)C4_dev_begins, {rowC}, {num_els}); CHECK_ERR;
    if (cuBlasStatus != CUBLAS_STATUS_SUCCESS) {{
        std::cout << "Second cuBlas call failed";
    }}
    //#ifndef NDEBUG
    //cudaDeviceSynchronize(); CHECK_ERR;
    //#endif
    CHECK_ERR;
    cudaEventRecord(stopcuSparse); CHECK_ERR;
    cudaEventSynchronize(stopcuSparse); CHECK_ERR;
    float elapsedTime4 = 0.0f;
    cudaEventElapsedTime(&elapsedTime4, startcuSparse, stopcuSparse); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;
    std::cout << "cuSparse DxS kernel took " << elapsedTime4 << " ms" << std::endl; 

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matBT) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matAT) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matCT) )
    CHECK_CUSPARSE( cusparseDestroy(cuSparseHandle) )

    float* RcuSparse = new float[{rowC} * {colC} * {num_els}];
    cudaMemcpy(RcuSparse, C4_dev, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost); CHECK_ERR;

    delete[] AT;
    delete[] CT;
    delete[] BT_data;
    delete[] BT_indices;
    delete[] BT_indptr;
    delete[] IdMat;
    delete[] C3_begins;
    delete[] C4_begins;
    cudaFree(C3_dev_begins); CHECK_ERR;
    cudaFree(C4_dev_begins); CHECK_ERR;
    delete[] IdMat_begins;
    cudaFree(IdMat_dev_begins); CHECK_ERR;
    cudaFree(IdMat_dev); CHECK_ERR;
    cudaFree(BT_data_dev); CHECK_ERR;
    cudaFree(BT_indices_dev); CHECK_ERR;
    cudaFree(BT_indptr_dev); CHECK_ERR;
    cudaFree(C3_dev); CHECK_ERR;
    cudaFree(dBuffer); CHECK_ERR;

    std::cout << "Freeing device memory" << std::endl;
    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] >= RcuSparse[i] * 1.001 || R1[i] <= RcuSparse[i] * 0.999) {{
            std::string s = "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse CuSparse Matrix Mismatch in Multiplication at " + std::to_string(i) + " ->" + 
                std::to_string(R1[i]) + " and " + std::to_string(RcuSparse[i]) + " | " + std::to_string(R1[{rowC}*{colC}*{num_els}-1]) + " and " + std::to_string(RcuSparse[{rowC}*{colC}*{num_els} - 1]);
            //throw std::runtime_error(s);
            std::cout << "RESULTS DONT MATCH" << std::endl;
            std::cout << s << std::endl;
            return 0;
        }}
    }}
    
    cudaFree(C4_dev); CHECK_ERR;
    delete[] RcuSparse;
    delete[] R1;
    delete[] RT;
    {f"cudaFree(B_sparse_dev); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaFree(A_dev);
    cudaFree(B_dense_dev);
    cudaFree(C1_dev);
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
