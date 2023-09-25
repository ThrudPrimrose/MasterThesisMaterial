import random
import sys
from random import randint

import numpy as np
from numba import cuda

from gemmforge import DenseMatrix, GenerationError, GemmGenerator, SparseMatrix
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType
from gemmforge.vm import vm_factory
from params import *
from scipy.sparse import csr_matrix, csc_matrix

def get_available_mem_on_gpu():
    gpus = cuda.gpus.lst
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]


def get_suggested_num_elements(MatBSize, MatADenseSize, MatASparseSize, MatCSize, SizeOfFloat):
    # We mul A x BD(dense) = C1, A x BS(Sparse) = C2
    # And compare C1 and C1, C1 and 2 obtained back will be R1 and R2 on host
    # On host we need A, BD, BS, C, R1, R2
    # On device we need A, BD, BS, C1, C2
    per_el_size = (MatBSize + MatADenseSize +
                   MatASparseSize + 2*MatCSize) * SizeOfFloat

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    at80 = int(0.60 * can_fit_els)
    return (can_fit_els, at80)


def gen_matrix_a(rowA, colA, transposed, atype):
    A = np.zeros([rowA, colA])
    coo = {"name": "A", "rows": rowA, "cols": colA,
           "entries": [], "coordinates": []}
    if atype == "full":
        for j in range(colA):
            for i in range(rowA):
                r1 = random.randint(1, 9)
                A[i, j] = r1
    elif atype == "band":
        block_size = min([rowA, colA])
        block_count = int(max([rowA, colA]) / block_size)
        block_direction = "x" if colA > rowA else "y"
        if block_direction == "y":
            for b in range(block_count):
                for i in range(block_size):
                    r1 = random.randint(1, 9)
                    A[i + block_size*b, i] = r1
                    if i > 0:
                        r2 = random.randint(1, 9)
                        A[i + block_size*b - 1, i] = r2
                    if i < block_size - 1:
                        r3 = random.randint(1, 9)
                        A[i + block_size*b + 1, i] = r3
            b = block_count
            j = 0
            for i in range(block_size*block_count, max([rowA, colA])):
                if i > block_size*block_count:
                    r1 = random.randint(1, 9)
                    A[i - 1, j] = r1
                r2 = random.randint(1, 9)
                A[i, j] = r2
                if i < max([rowA, colA]) - 1:
                    r3 = random.randint(1, 9)
                    A[i + 1, j] = r3
                j += 1
        elif block_direction == "x":
            for b in range(block_count):
                for i in range(block_size):
                    r1 = random.randint(1, 9)
                    A[i, i + block_size*b] = r1
                    if i > 0:
                        r2 = random.randint(1, 9)
                        A[i - 1, i + block_size*b ] = r2
                    if i < block_size - 1:
                        r3 = random.randint(1, 9)
                        A[i + 1, i + block_size*b ] = r3
            b = block_count
            i = 0
            for j in range(block_size*block_count, max([rowA, colA])):
                if i > 0:
                    r1 = random.randint(1, 9)
                    A[i - 1, j] = r1
                r2 = random.randint(1, 9)
                A[i, j] = r2
                if i < max([rowA, colA]) - 1:
                    r3 = random.randint(1, 9)
                    A[i + 1, j] = r3
                i += 1
    elif atype == "random":
        entry_count = int(non_zero_ratio * rowA * colA)
        a_el_count = entry_count
        l = set()
        while len(l) < entry_count:
            i = randint(0, rowA - 1)
            j = randint(0, colA - 1)
            l.add((i, j))
        llist = list(l)
        assert (len(llist) == a_el_count)
        for (row, col) in llist:
            A[row, col] = 1
        for j in range(colA):
            for i in range(rowA):
                if A[i, j] != 0:
                    r = random.randint(1, 9)
                    A[i, j] = r
    elif atype == "chequered":
        for i in range(rowA):
            for j in range(colA):
                if i % 2 == 0:
                    if j % 2 == 0:
                        r1 = random.randint(1, 9)
                        A[i, j] = r1
                else:
                    if j % 2 == 1:
                        r2 = random.randint(1, 9)
                        A[i, j] = r2
    else:
        raise Exception("NO")

    if transposed:
        npA = A.T
    else:
        npA = A
    A = npA.flatten("F")
    A_nonzeros = []
    i = 0
    #print(npA)
    #print(A)
    for el in A:
        if el > 0.00001 or el < -0.00001:
            assert (el != 0 and el != 0.0)
            A_nonzeros.append(el)
            coords = np.unravel_index(i, (rowA, colA), "F")
            print(coords)

            #if transposed:
            #    coo["coordinates"].append([coords[1], coords[0]])
            #    coo["entries"].append([coords[1], coords[0], el])
            #else:
            coo["coordinates"].append([coords[0], coords[1]])
            coo["entries"].append([coords[0], coords[1], el])
        i += 1
    a_el_count = len(coo["coordinates"])
    return (coo, A, A_nonzeros, a_el_count, npA)


try:
    for with_compile_time_values in [False, True]:
        for a_type in a_matrix_types:
            for tA in [False, True]:
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
                    print(f"Generating {testid}")
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
                        rowB = col_b
                        colB = row_b
                    rowC = row_c
                    colC = col_c

                    coo, matrix_a, matrix_a_non_zeros_flat, a_el_count, npA = gen_matrix_a(
                        rowA, colA, tA, a_type)
                    #print(npA)
                    #print(matrix_a_non_zeros_flat)
                    #print(coo["coordinates"])

                    mat_a_dense = DenseMatrix(num_rows=rowA,
                                              num_cols=colA,
                                              addressing="strided",
                                              bbox=[0, 0, rowA, colA])

                    mat_a_sparse = SparseMatrix(num_rows=rowA,
                                                num_cols=colA,
                                                addressing="strided",
                                                coordinates=coo["coordinates"],
                                                values=matrix_a_non_zeros_flat if with_compile_time_values else None)

                    #if tA:
                    #    mat_a_csr = csr_matrix(npA.T)
                    #else:
                    mat_a_csr = csr_matrix(npA)
                    mat_a_csr.sort_indices()
                    A_data = mat_a_csr.data
                    A_indices = mat_a_csr.indices
                    A_indptr = mat_a_csr.indptr
                    strA_data = "{" + ", ".join([str(x) for x in A_data]) + "}"
                    strA_indices = "{" + ", ".join([str(x) for x in A_indices]) + "}"
                    strA_indptr = "{" + ", ".join([str(x) for x in A_indptr]) + "}"

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

                    T = "t"
                    NT = ""
                    dense_gen = GemmGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    dense_gen.set(tA, tB, mat_a_dense, mat_b, mat_c, alpha=Alpha, beta=Beta,
                                  base_name=f"A{T if transA else NT}_B{T if transB else NT}_DenseXDense")
                    dense_gen.generate()
                    dense_header = dense_gen.get_launcher_header()
                    dense_function_name = dense_header.split("(")[0][4:]

                    sparse_gen = GemmGenerator(
                        vm=vm, kernel_type=GemmKernelType.AUTO)
                    sparse_gen.set(tA, tB, mat_a_sparse, mat_b, mat_c, alpha=Alpha, beta=Beta,
                                   base_name=f"A{T if transA else NT}_{a_type}_B{T if transB else NT}_SparseXDense")
                    sparse_gen.generate()
                    sparse_header = sparse_gen.get_launcher_header()
                    sparse_function_name = sparse_header.split("(")[0][4:]

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
                    strB = np.array2string(B, separator=', ', max_line_width=10000).replace(
                        "[", "{").replace("]", "}")
                    strA_sparse = np.array2string(np.array(A_sparse), separator=', ', max_line_width=10000).replace(
                        "[", "{").replace("]", "}")
                    strA_dense = np.array2string(A_dense, separator=', ', max_line_width=10000).replace(
                        "[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ', max_line_width=10000).replace(
                        "[", "{").replace("]", "}")

                    get_available_mem_on_gpu()
                    full, at80 = get_suggested_num_elements(
                        rowB * colB, rowA * colA, a_el_count, rowC * colC, 4)
                    num_els = at80

                    ctv = "_ctv" if with_compile_time_values else ""
                    n = f"A{T if transA else NT}_{a_type}_B{T if transB else NT}_SparseXDense{ctv}"
                    s = f"""
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <cublas_v2.h>
#include <cusparse.h>

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
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
        printf("CUSPARSE API failed at line %d with error: %s (%d)\\n",        \\
               __LINE__, cusparseGetErrorString(status), status);              \\
        throw std::runtime_error("CUSPARSE API failed");                       \\
    }}                                                                         \\
}}

#define CHECK_CUBLAS(func)                                                     \\
{{                                                                             \\
    cublasStatus_t status = (func);                                            \\
    if (status != CUBLAS_STATUS_SUCCESS) {{                                    \\
        printf("CUBLAS API failed at line %d with error: %s (%d)\\n",          \\
               __LINE__, cublasGetStatusString(status), status);               \\
        throw std::runtime_error("CUBLAS API failed");                         \\
    }}                                                                         \\
}}

#define CHECK_CUDA(func)                                                       \\
{{                                                                             \\
    cudaError_t status = (func);                                               \\
    if (status != cudaSuccess) {{                                              \\
        printf("CUDA API failed at line %d with error: %s (%d)\\n",            \\
               __LINE__, cudaGetErrorString(status), status);                  \\
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


int main(){{
  float *alpha_dev = nullptr;
  float *beta_dev = nullptr;
  float *transpose_alpha_dev = nullptr;
  float *transpose_beta_dev = nullptr;
  float alpha = {Alpha}f;
  float beta = {Beta}f;
  float talpha = 1.0f;
  float tbeta = 0.0f;
  cudaMalloc(&alpha_dev, sizeof(float));
  cudaMalloc(&beta_dev, sizeof(float));
  cudaMalloc(&transpose_alpha_dev, sizeof(float));
  cudaMalloc(&transpose_beta_dev, sizeof(float));
  cudaMemcpy((void*)alpha_dev, (void*)&alpha, sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void*)beta_dev, (void*)&beta, sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void*)transpose_alpha_dev, (void*)&talpha, sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void*)transpose_beta_dev, (void*)&tbeta, sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

  cublasHandle_t handle;
  CHECK_CUBLAS( cublasCreate(&handle) )
  CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) )

  int num_els = {num_els};
  if ({num_els}>std::numeric_limits<int>::max()){{
    throw std::runtime_error("Batch size too huge for num_els");
  }}
  constexpr int cuSparseBatchSize = 65535;
  constexpr int cudaStreamsNeeded = {num_els / 65535 + int(num_els % 65535!= 0)};
  cudaStream_t streams[cudaStreamsNeeded];
  for (int i = 0; i < cudaStreamsNeeded; i++) {{
      cudaStreamCreate(&streams[i]);
  }}

  std::cout << "Gemm-Type: " << "{n}" << std::endl;
  std::cout << "Number of elements: " << num_els << std::endl;
  // Element Matrices
  std::cout << "Instantiating core matrices" << std::endl;
  float CoreA_sparse[{a_el_count}] = {strA_sparse};
  float CoreA_dense[{rowA}*{colA}] = {strA_dense};
  float CoreB[{rowB} * {colB}] = {strB};
  float CoreC[{rowC}*{colC}] = {strC};
  float CoreA_data[{len(A_data)}] = {strA_data};
  int CoreA_indices[{len(A_indices)}] = {strA_indices};
  int CoreA_indptr[{len(A_indptr)}] = {strA_indptr};

  // Buffers 
  std::cout << "Instantiating buffer matrices" << std::endl;
  float* A_dense = new float[{rowA}*{colA}*num_els];
  {f"float* A_sparse = new float[{a_el_count}*num_els];" if not with_compile_time_values else ""}
  float* B = new float[{rowB}*{colB}*num_els];
  float* C = new float[{rowC}*{colC}*num_els];
  float* R1 = new float[{rowC}*{colC}*num_els];
  float* R2 = new float[{rowC}*{colC}*num_els];
  float* A_data = new float[{len(A_data)}*num_els];
  float* A_indices = new float[{len(A_indices)}*num_els];
  float* A_indptr = new float[{len(A_indptr)}*num_els];

  // Copy the Element Matrices N times into Element Buffers
  std::cout << "Copying core matrices to buffers" << std::endl;
  for (int i = 0; i < num_els; i++){{
      std::memcpy(&A_dense[{rowA} * {colA} * i], &CoreA_dense[0], {rowA} * {colA} * sizeof(float));
      std::memcpy(&B[{rowB} * {colB} * i], &CoreB[0], {rowB} * {colB} * sizeof(float));
      {f"std::memcpy(&A_sparse[{a_el_count} * i], &CoreA_sparse[0], {a_el_count} * sizeof(float));" if not with_compile_time_values else ""}
      std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
      std::memcpy(&A_data[{len(A_data)} * i], &CoreA_data[0], {len(A_data)} * sizeof(float));
      std::memcpy(&A_indices[{len(A_indices)} * i], &CoreA_indices[0], {len(A_indices)} * sizeof(int));
      std::memcpy(&A_indptr[{len(A_indptr)} * i], &CoreA_indptr[0], {len(A_indptr)} * sizeof(int));
  }}

  float *A_dense_dev = nullptr;
  {"float *A_sparse_dev = nullptr;" if not with_compile_time_values else ""}
  float *B_dev = nullptr;
  float *C_dev = nullptr;

  std::cout << "Allocating device memory" << std::endl;
  cudaMalloc((void **)&B_dev, sizeof(float) * {rowB} * {colB} * num_els); CHECK_ERR;
  {f"cudaMalloc((void **)&A_sparse_dev, sizeof(float) * {a_el_count} * num_els); CHECK_ERR;" if not with_compile_time_values else ""}
  cudaMalloc((void **)&A_dense_dev, sizeof(float) * {rowA} * {colA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {rowC} * {colC} * num_els); CHECK_ERR;

  std::cout << "Copying buffers to device" << std::endl;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {rowB} * {colB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  {f"cudaMemcpy((void *)A_sparse_dev, (void *)A_sparse, sizeof(float) *  {a_el_count} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;" if not with_compile_time_values else ""}
  cudaMemcpy((void *)A_dense_dev, (void *)A_dense, sizeof(float) *  {rowA} * {colA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {dense_function_name}(A_dense_dev, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  // Dense x Dense Matrix Mult
  std::cout << "Calling Dense x Dense kernel" << std::endl;
  float elapsedTime = 0.0; 
  cudaEvent_t startDD, stopDD;
  cudaEventCreate(&startDD); CHECK_ERR;
  cudaEventCreate(&stopDD); CHECK_ERR;
  cudaEventRecord(startDD); CHECK_ERR;
  {dense_function_name}(A_dense_dev, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopDD); CHECK_ERR;
  cudaEventSynchronize(stopDD); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDD, stopDD); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Dense x Dense kernel took " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, C_dev, sizeof(float)*{rowC} * {colC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {f"{sparse_function_name}(A_sparse_dev, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(nullptr, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr);"} CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  // Sparse x Dense Matrix Mult
  std::cout << "Calling Sparse x Dense kernel" << std::endl;
  elapsedTime = 0.0;
  cudaEvent_t startDS, stopDS;
  cudaEventCreate(&startDS); CHECK_ERR;
  cudaEventCreate(&stopDS); CHECK_ERR;
  cudaEventRecord(startDS); CHECK_ERR;
  {f"{sparse_function_name}(A_sparse_dev, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(nullptr, 0, B_dev, 0, C_dev, 0, num_els, nullptr, nullptr);"} CHECK_ERR;
  cudaEventRecord(stopDS); CHECK_ERR;
  cudaEventSynchronize(stopDS); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Sparse x Dense kernel took " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R2, C_dev, sizeof(float)*{rowC} * {colC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  bool wrong_results = false;
  for (int i = 0; i < {rowC}*{colC}*num_els; i++){{
    if (std::abs(R1[i] - R2[i]) >= 0.1) {{
      std::string s = "{transA} Dense x {transB} Dense and {transA} Sparse x {transB} Matrix Mismatch in Multiplication at " + std::to_string(i) + "!";
      std::cout << " " << std::to_string(R1[i]) + " and " + std::to_string(R2[i]) << std::endl;
      std::cout << "Dense x Dense Gemmforge and Dense x Sparse Gemmforge results don't match!" << std::endl;
      std::cout << s << std::endl;
      wrong_results = true;
      break;
    }}
  }}

  if (!wrong_results){{
    std::cout << "{transA} Dense x {transB} Dense and {transA} Sparse x {transB} Dense Matrix Multiplications Match!" << std::endl;
  }}

  std::cout << "Calling cuBlas DxD Kernels" << std::endl;
  float elapsedTime3 = 0.0;
  float** A_dense_begins = new float*[num_els];
  float** B_begins = new float*[num_els];
  float** C_begins = new float*[num_els];
  float** A_dense_dev_begins = nullptr;
  float** B_dev_begins = nullptr;
  float** C_dev_begins = nullptr;
  for (int i = 0; i < num_els; i++){{
      A_dense_begins[i]  = &A_dense_dev[0]  + {rowA} * {colA} * i;
      B_begins[i]  = &B_dev[0]  + {rowB} * {colB} * i;
      C_begins[i] = &C_dev[0] + {rowC} * {colC} * i;
  }}
  cudaMalloc((void **)&A_dense_dev_begins, num_els * sizeof(float*)); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins, num_els * sizeof(float*)); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins, num_els * sizeof(float*)); CHECK_ERR;
  cudaMemcpy((void *)A_dense_dev_begins, (void *)A_dense_begins, num_els * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins, (void *)B_begins, num_els * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins, (void *)C_begins, num_els * sizeof(float*), cudaMemcpyHostToDevice); CHECK_ERR;
  // First 2 to discard
  cublasStatus_t cuBlasStatus = cublasSgemmBatched(handle, {"CUBLAS_OP_T" if transA else "CUBLAS_OP_N"}, {"CUBLAS_OP_T" if transB else "CUBLAS_OP_N"},
      {rowC}, {colC}, {rowB}, alpha_dev, (const float **)A_dense_dev_begins, {rowA}, (const float **)B_dev_begins, {rowB},
      beta_dev, (float **)C_dev_begins, {rowC}, num_els); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  if (cuBlasStatus != CUBLAS_STATUS_SUCCESS) {{
      std::cout << "First cuBlas call failed";
  }}
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  cudaEvent_t startCuBlas, stopCuBlas; 
  cudaEventCreate(&startCuBlas); CHECK_ERR;
  cudaEventCreate(&stopCuBlas); CHECK_ERR;
  cudaEventRecord(startCuBlas); CHECK_ERR;
  cuBlasStatus = cublasSgemmBatched(handle, {"CUBLAS_OP_T" if transA else "CUBLAS_OP_N"}, {"CUBLAS_OP_T" if transB else "CUBLAS_OP_N"},
      {rowC}, {colC}, {rowB}, (const float*)alpha_dev, (const float **)A_dense_dev_begins, {rowA}, (const float **)B_dev_begins, {rowB},
      (const float*)beta_dev, (float **)C_dev_begins, {rowC}, num_els); CHECK_ERR;
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
  cudaMemcpy(R2, C_dev, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  bool wrong_cuBlas_results = false;
  for (int i = 0; i < {rowC}*{colC}*num_els; i++){{
    if (std::abs(R1[i] - R2[i]) >= 0.1) {{
      std::string s = "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + " ->" + 
        std::to_string(R1[i]) + " and " + std::to_string(R2[i]) + " | " + std::to_string(R1[{rowC}*{colC}*num_els-1]) + " and " + std::to_string(R2[{rowC}*{colC}*num_els - 1]);
      std::cout << "Gemmforge and cuBlas results don't match" << std::endl;
      std::cout << s << std::endl;
      wrong_cuBlas_results = true;
      break;
    }}
  }}
  
  if (!wrong_cuBlas_results){{
    std::cout << "Gemmforge Dense x Dense and cuBlas Dense x Dense results match" << std::endl;
  }}

  std::cout << "Freeing device-memory needed by cuBLAS-only" << std::endl;

  delete[] A_dense_begins;
  delete[] B_begins;
  delete[] C_begins;
  cudaFree(A_dense_dev_begins); CHECK_ERR;
  cudaFree(C_dev_begins); CHECK_ERR;
  cudaFree(B_dev_begins); CHECK_ERR;

  std::cout << "Calling cuSparse SxD Kernels" << std::endl;
  cusparseHandle_t cuSparseHandle;
  CHECK_CUSPARSE( cusparseCreate(&cuSparseHandle) )

  cusparseSpMatDescr_t cuA[cudaStreamsNeeded];
  cusparseDnMatDescr_t cuB[cudaStreamsNeeded];
  cusparseDnMatDescr_t cuC[cudaStreamsNeeded];
  void** dBuffers    = new void*[cudaStreamsNeeded];
  float* A_data_dev  = nullptr;
  int* A_indices_dev = nullptr;
  int* A_indptr_dev  = nullptr;
  cudaMalloc((void **)&A_data_dev, num_els * {len(A_data)} * sizeof(float)); CHECK_ERR;
  cudaMalloc((void **)&A_indices_dev, num_els * {len(A_indices)} * sizeof(int)); CHECK_ERR;
  cudaMalloc((void **)&A_indptr_dev, num_els * {len(A_indptr)} * sizeof(int)); CHECK_ERR;
  cudaMemcpy((void *)A_data_dev, (void *)A_data, sizeof(float) *  {len(A_data)} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)A_indices_dev, (void *)A_indices, sizeof(int) * {len(A_indices)} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)A_indptr_dev, (void *)A_indptr, sizeof(int) * {len(A_indptr)} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void* )C_dev, (void *)C, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void* )B_dev, (void *)B, sizeof(float) * {rowB} * {colB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (int i =0; i<cudaStreamsNeeded; i++){{
    int cuSparse_num_els = cuSparseBatchSize;
    if (i == cudaStreamsNeeded - 1){{
        cuSparse_num_els = num_els - i*cuSparseBatchSize;
    }}
    CHECK_CUSPARSE( cusparseCreateCsr(&cuA[i], {rowA}, {colA}, {len(A_data)},
                                        A_indptr_dev + i * cuSparseBatchSize * {len(A_indptr)},
                                        A_indices_dev + i * cuSparseBatchSize * {len(A_indices)},
                                        A_data_dev + i * cuSparseBatchSize * {len(A_data)},
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCsrSetStridedBatch(cuA[i], cuSparse_num_els, {len(A_indptr)}, {len(A_data)}))

    CHECK_CUSPARSE( cusparseCreateDnMat(&cuB[i], {rowB}, {colB}, {rowB},
                                        B_dev + i * cuSparseBatchSize * {rowB} * {colB},
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(cuB[i], cuSparse_num_els, {rowB} * {colB}) )

    CHECK_CUSPARSE( cusparseCreateDnMat(&cuC[i], {rowC}, {colC}, {rowC}, 
                                        C_dev + i * cuSparseBatchSize * {rowC} * {colC},
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(cuC[i], cuSparse_num_els, {rowC} * {colC}) )

    size_t               bufferSize = 0;
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                    cuSparseHandle,
                                    {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not tA else "CUSPARSE_OPERATION_TRANSPOSE"},
                                    {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not tB else "CUSPARSE_OPERATION_TRANSPOSE"},
                                    &alpha, cuA[i], cuB[i], &beta, cuC[i], CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
  }}
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaEvent_t startcuSparse, stopcuSparse;
  cudaEventCreate(&startcuSparse); CHECK_ERR;
  cudaEventCreate(&stopcuSparse); CHECK_ERR;
  cudaEventRecord(startcuSparse); CHECK_ERR;
  for (int i =0; i<cudaStreamsNeeded; i++){{
    cusparseSetStream(cuSparseHandle, streams[i]);
    CHECK_CUSPARSE( cusparseSpMM(cuSparseHandle,
                                    {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not tA else "CUSPARSE_OPERATION_TRANSPOSE"},
                                    {"CUSPARSE_OPERATION_NON_TRANSPOSE" if not tB else "CUSPARSE_OPERATION_TRANSPOSE"},
                                    (const void*)&alpha, cuA[i], cuB[i], (const void*)&beta, cuC[i], CUDA_R_32F,
                                    CUSPARSE_SPMM_CSR_ALG2, (void*)dBuffers[i]) )
  }}
  cudaEventRecord(stopcuSparse); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  float elapsedTime4 = 0.0f;
  cudaEventElapsedTime(&elapsedTime4, startcuSparse, stopcuSparse); CHECK_ERR;
  std::cout << "cuSparse SxD kernel took " << elapsedTime4 << " ms" << std::endl; 

  cudaMemcpy(R2, C_dev, sizeof(float) * {rowC} * {colC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  bool cuSparseWrong = false;
  for (int i = 0; i < {rowC}*{colC}*num_els; i++){{
      if (std::abs(R1[i] - R2[i]) >= 0.1) {{
          std::string s = "{transA} Dense x {transB} Dense and {transA} Sparse x {transB} Dense CuSparse Matrix Mismatch in Multiplication at " + std::to_string(i) + " ->" + 
              std::to_string(R1[i]) + " and " + std::to_string(R2[i]) + " | " + std::to_string(R1[{rowC}*{colC}*num_els-1]) + " and " + std::to_string(R2[{rowC}*{colC}*num_els - 1]);
          std::cout << "Gemmforge Dense x Dense and cuSparse RESULTS DONT MATCH" << std::endl;
          std::cout << s << std::endl;
          cuSparseWrong = true;
          break;
      }}
  }}
  if (!cuSparseWrong){{
    std::cout << "Gemmforge Dense x Dense and cuSparse Sparse x Dense results match!" << std::endl;
  }}

  std::cout << "Freeing device memory" << std::endl;

  for (int i = 0; i < cudaStreamsNeeded; i++){{
    CHECK_CUSPARSE( cusparseDestroySpMat(cuA[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(cuB[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(cuC[i]) )
  }}

  CHECK_CUSPARSE( cusparseDestroy(cuSparseHandle) )

  cudaFree(A_dense_dev);
  {f"cudaFree(A_sparse_dev);" if not with_compile_time_values else ""}
  cudaFree(B_dev);
  cudaFree(C_dev);
  cudaFree(A_data_dev);
  cudaFree(A_indptr_dev);
  cudaFree(A_indices_dev);
  cudaFree(alpha_dev);
  cudaFree(beta_dev);
  cudaFree(transpose_alpha_dev);
  cudaFree(transpose_beta_dev);
  for (int i = 0; i < cudaStreamsNeeded; i++){{
      cudaFree(dBuffers[i]);
  }}

  delete[] A_dense;
  {f"delete[] A_sparse;" if not with_compile_time_values else ""}
  delete[] B;
  delete[] C;
  delete[] R1;
  delete[] R2;
  delete[] A_data;
  delete[] A_indices;
  delete[] A_indptr;
  delete[] dBuffers;

  for (int i = 0; i < cudaStreamsNeeded; i++) {{
      cudaStreamDestroy(streams[i]);
  }}

  cudaDeviceReset();
}}
"""
                    f = open(
                        f"{scripts_dir}/cuda_code/benchmark_cuda_sparse_dense_{testid}.cu", "w")
                    f.write(s)
                    f.close()
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
