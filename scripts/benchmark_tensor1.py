import os
from functools import reduce
import operator

from yateto import *
from tensor_params import *
import numpy as np
from numba import cuda

N = 32
shapeA = (N, N)
shapeB = (N, N, N)
shapew = (N, )
shapeC = (N, N)
A = Tensor('A', shapeA)
B = Tensor('B', shapeB)
w = Tensor('w', shapew)
C = Tensor('C', shapeC)
sizeA = reduce(operator.mul, shapeA, 1)
sizeB = reduce(operator.mul, shapeB, 1)
sizew = reduce(operator.mul, shapew, 1)
sizeC = reduce(operator.mul, shapeC, 1)

def get_available_mem_on_gpu():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]

def get_suggested_num_elements():
    #1 pointer extra needed per element
    per_el_size = (sizeA + sizeB + sizew + sizeC) * 4 + 16

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    lower = int(0.85 * can_fit_els)
    return lower

num_els = get_suggested_num_elements()

kernel = C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k']

arch = useArchitectureIdentifiedBy(
    host_arch="shsw", device_arch="ssm_86", device_backend="cuda")
generator = Generator(arch)

generator.add(name='kernel', ast=kernel, target="gpu")

directory = os.path.dirname(os.path.abspath(__file__))
generator.generate(outputDir="/tmp",
                   gemm_cfg=GeneratorCollection([GemmForge(arch), Eigen(arch)]))

assert(len(generator.kernels()) == 1)

gpu_kernel = generator.kernels()[0]
gpu_subroutine_file = open("/tmp/gpulike_subroutine.cpp", "r")
gpu_kernel = gpu_subroutine_file.read()
gpu_subroutine_file.close()

gemmforge_kernel_lines = gpu_kernel.split('\n')

function_name = ""
function_args = ""

filtered_kernel = ""
for line in gemmforge_kernel_lines:
  if '#include "gemmforge_aux.h"' in line:
    continue
  filtered_kernel += line + "\n"
  if line.startswith("void sloopOverGEMM"):
    fun_split = line.split("(")
    function_name = fun_split[0].split("void ")[1]
    function_args = fun_split[1].split(")")[0]

gpu_kernel = filtered_kernel

print(function_name)
print(function_args)

a = A.memoryLayout
print(a)

benchmark_str = f"""
#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

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

{gpu_kernel}

int main(){{
  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els];
  float* B = new float[{sizeB} * num_els];
  float* C = new float[{sizeC} * num_els];
  float* w = new float[{sizew} * num_els];
  float* R = new float[{sizeC} * num_els]{{0.f}};

  float coreA[{sizeA}];
  float coreB[{sizeB}];
  float coreC[{sizeC}];
  float corew[{sizew}];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < {sizeA}; i++){{
    coreA[i] = distribution(gen);
  }}
  for (size_t i = 0; i < {sizeB}; i++){{
    coreB[i] = distribution(gen);
  }}
  for (size_t i = 0; i < {sizeC}; i++){{
    coreC[i] = distribution(gen);
  }}
  for (size_t i = 0; i < {sizew}; i++){{
    corew[i] = distribution(gen);
  }}

  for (size_t i = 0; i < num_els; i++){{
      std::memcpy(&A[i * {sizeA}], &coreA[0], {sizeA} * sizeof(float));
      std::memcpy(&B[i * {sizeB}], &coreB[0], {sizeB} * sizeof(float));
      std::memcpy(&C[i * {sizeC}], &coreC[0], {sizeC} * sizeof(float));
      std::memcpy(&w[i * {sizew}], &corew[0], {sizew} * sizeof(float));
  }}

  float* A_dev = nullptr;
  float* B_dev = nullptr;
  float* C_dev = nullptr;
  float* w_dev = nullptr;

  float** A_dev_begins = new float*[num_els];
  float** B_dev_begins = new float*[num_els];
  float** C_dev_begins = new float*[num_els];
  float** w_dev_begins = new float*[num_els];

  float** A_dev_begins_dev = nullptr;
  float** B_dev_begins_dev = nullptr;
  float** C_dev_begins_dev = nullptr;
  float** w_dev_begins_dev = nullptr;

  std::cout << "Allocating device memory" << std::endl;
  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev, sizeof(float) * {sizew} * num_els); CHECK_ERR;
  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;

  std::cout << "Copying buffers to device" << std::endl;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {sizeB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev, (void *)w, sizeof(float) * {sizew} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{
    A_dev_begins[i] = A_dev + i * {sizeA};
    B_dev_begins[i] = B_dev + i * {sizeB};
    C_dev_begins[i] = C_dev + i * {sizeC};
    w_dev_begins[i] = w_dev + i * {sizew};
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev_begins_dev, (void *)w_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {function_name}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  float elapsedTime = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  {function_name}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startT1, stopT1); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  cudaFree(w_dev);
  cudaFree(A_dev_begins_dev);
  cudaFree(B_dev_begins_dev);
  cudaFree(C_dev_begins_dev);
  cudaFree(w_dev_begins_dev);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] w;
  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;
  delete[] w_dev_begins;

  return 0;
}}

"""

code_file = open(f"{scripts_dir}/cuda_code/benchmark_tensor1.cu", "w")
code_file.write(benchmark_str)
code_file.flush()
code_file.close()