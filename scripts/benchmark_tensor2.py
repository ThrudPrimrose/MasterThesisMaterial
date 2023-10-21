import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda

N = 32
shapeA = (N, N, N)
shapeB = (N, )
shapeC = (N, N)
shapeD = (N, N)
shapeE = (N, N, N)
shapeF = (N, N)
A = Tensor('A', shapeA)
B = Tensor('B', shapeB)
C = Tensor('C', shapeC)
D = Tensor('D', shapeD)
E = Tensor('E', shapeE)
F = Tensor('F', shapeF)
sizeA = reduce(operator.mul, shapeA, 1)
sizeB = reduce(operator.mul, shapeB, 1)
sizeC = reduce(operator.mul, shapeC, 1)
sizeD = reduce(operator.mul, shapeD, 1)
sizeE = reduce(operator.mul, shapeE, 1)
sizeF = reduce(operator.mul, shapeF, 1)
maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC), 
                  len(shapeD), len(shapeE), len(shapeF))

def get_available_mem_on_gpu():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]


def get_suggested_num_elements():
    # 1 pointer extra needed per element
    per_el_size = (sizeA + sizeB + sizeC + sizeD + sizeE + sizeF) * 4 + (6 * 4)

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    lower = int(0.90 * can_fit_els)
    return lower


num_els = get_suggested_num_elements()

# Taken from addLocal viscoelastic2.py
kernel = A['kpm'] <= A['kpm'] + B['m'] * C['kq'] * D['qp'] + E['kpl'] * F['lm']

arch = useArchitectureIdentifiedBy(
    host_arch="shsw", device_arch="ssm_86", device_backend="cuda")
generator = Generator(arch)

generator.add(name='kernel', ast=kernel, target="gpu")

directory = os.path.dirname(os.path.abspath(__file__))
generator.generate(outputDir="/tmp",
                   gemm_cfg=GeneratorCollection([GemmForge(arch), Eigen(arch)]))

assert (len(generator.kernels()) == 1)

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

"""
taco_command = \
  f'taco "C(i, j, b) = C(i, j, b) + A(l, j, b) * B(i, k, l, b) * w(k, b)" -cuda -d=A:{shapeA[0]},{shapeA[1]},{num_els} -d=B:{shapeB[0]},{shapeB[1]},{shapeB[2]},{num_els} -d=C:{shapeC[0]},{shapeC[1]},{num_els} -d=w:{shapew[0]},{num_els} -t=A:float -t=B:float -t=C:float -t=w:float -print-nocolor '

print(taco_command)

result = subprocess.run(taco_command, shell=True, stdout=subprocess.PIPE, text=True)
taco_kernel = result.stdout

taco_kernel_lines = taco_kernel.split("\n")
taco_kernel = ""
launcher_line = 99999999999999
for i, line in enumerate(taco_kernel_lines):
  if "int compute" in line:
    args = line.split("(")[1].split(")")[0]
    args = args.split(",")
    args = [s.strip() for s in args]
    args_unique = sorted(list(set(args)))
    taco_kernel += "int compute(" + ", ".join(args_unique) + ") {" + "\n"
    launcher_line = i
    continue
  elif "_dimension" in line and i > launcher_line:
    launcher_line = 9999999999999999
    assert("->dimensions" in line)
    tensorName = line.split("->dimensions")[0][-1]
    offset = int(line.split("->dimensions")[1][1])
    if tensorName == "A":
      dim = shapeA[offset]
    elif tensorName == "B":
      dim = shapeB[offset]
    elif tensorName == "C":
      dim = shapeC[offset]
    elif tensorName == "w":
      dim = shapew[offset]
    line_f = line.split("=")[0]
    line_f += " = " + str(dim) + ";\n"
    taco_kernel += line_f
    continue 
  else:
    taco_kernel += line + "\n"
    continue
"""

benchmark_str = f"""
#include <random>
#include <iostream>
#include <cstring>
#include <vector>
#include <unordered_map>

#include <cutensor.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(x)                                                  \\
{{                                                                        \\
  const auto err = x;                                                    \\
  if( err != CUTENSOR_STATUS_SUCCESS )                                   \\
  {{                                                                      \\
    std::cout << "Error: " << cutensorGetErrorString(err) << std::endl;  \\
    return err;                                                          \\
  }}                                                                      \\
}}

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

struct taco_tensor_t {{
  float* vals;
  int* dimensions;
}};


int main(){{
  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els];
  float* B = new float[{sizeB} * num_els];
  float* C = new float[{sizeC} * num_els];
  float* D = new float[{sizeD} * num_els];
  float* E = new float[{sizeE} * num_els];
  float* F = new float[{sizeF} * num_els];
  float* R1 = new float[{sizeC} * num_els]{{0.f}};

  float coreA[{sizeA}];
  float coreB[{sizeB}];
  float coreC[{sizeC}];
  float coreD[{sizeD}];
  float coreE[{sizeE}];
  float coreF[{sizeF}];

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
  for (size_t i = 0; i < {sizeD}; i++){{
    coreD[i] = distribution(gen);
  }}
  for (size_t i = 0; i < {sizeE}; i++){{
    coreE[i] = distribution(gen);
  }}
  for (size_t i = 0; i < {sizeF}; i++){{
    coreF[i] = distribution(gen);
  }}

  for (size_t i = 0; i < num_els; i++){{
      std::memcpy(&A[i * {sizeA}], &coreA[0], {sizeA} * sizeof(float));
      std::memcpy(&B[i * {sizeB}], &coreB[0], {sizeB} * sizeof(float));
      std::memcpy(&C[i * {sizeC}], &coreC[0], {sizeC} * sizeof(float));
      std::memcpy(&D[i * {sizeD}], &coreD[0], {sizeD} * sizeof(float));
      std::memcpy(&E[i * {sizeE}], &coreE[0], {sizeE} * sizeof(float));
      std::memcpy(&F[i * {sizeF}], &coreF[0], {sizeF} * sizeof(float));
  }}

  float* A_dev = nullptr;
  float* B_dev = nullptr;
  float* C_dev = nullptr;
  float* D_dev = nullptr;
  float* E_dev = nullptr;
  float* F_dev = nullptr;

  float** A_dev_begins = new float*[num_els];
  float** B_dev_begins = new float*[num_els];
  float** C_dev_begins = new float*[num_els];
  float** D_dev_begins = new float*[num_els];
  float** E_dev_begins = new float*[num_els];
  float** F_dev_begins = new float*[num_els];

  float** A_dev_begins_dev = nullptr;
  float** B_dev_begins_dev = nullptr;
  float** C_dev_begins_dev = nullptr;
  float** D_dev_begins_dev = nullptr;
  float** E_dev_begins_dev = nullptr;
  float** F_dev_begins_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * {sizeD} * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * {sizeE} * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * {sizeF} * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
 
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {sizeB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * {sizeD} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * {sizeE} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * {sizeF} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{
    A_dev_begins[i] = A_dev + i * {sizeA};
    B_dev_begins[i] = B_dev + i * {sizeB};
    C_dev_begins[i] = C_dev + i * {sizeC};
    D_dev_begins[i] = D_dev + i * {sizeD};
    E_dev_begins[i] = E_dev + i * {sizeE};
    F_dev_begins[i] = F_dev + i * {sizeF};
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {function_name}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, D_dev_begins_dev, 0, E_dev_begins_dev, 0, F_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Will compute the kernel: A['kpm'] <= A['kpm'] + B['m'] * C['kq'] * D['qp'] + E['kpl'] * F['lm'], with Gemmforge" << std::endl;
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
  cudaMemcpy(R1, A_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  delete[] E;
  delete[] F;
  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;
  delete[] D_dev_begins;
  delete[] E_dev_begins;
  delete[] F_dev_begins;
  delete[] R1;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  cudaFree(D_dev);
  cudaFree(E_dev);
  cudaFree(F_dev);

  return 0;
}}

"""

code_file = open(f"{scripts_dir}/cuda_code/benchmark_tensor2.cu", "w")
code_file.write(benchmark_str)
code_file.flush()
code_file.close()
