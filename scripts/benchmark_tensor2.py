import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda
import random

"""
dims = [(32,32,32,32,32),
        (3,11,5,7,13),
]

shrmem_limit = 48*1024
while len(dims) < 50:
  k = random.randint(8, 100)
  p = random.randint(8, 100)
  m = random.randint(8, 100)
  q = random.randint(8, 100)
  l = random.randint(8, 100)
  #Loaded at the same time: B, X -> M + K*P
  #F -> L*M
  #D -> Q*P
  mem_needed = max(m + k*p, l*m, q*p)
  max_thread = max(m*k, k*p)
  if mem_needed * 4 > shrmem_limit:
    continue
  if max_thread > 1024:
    continue
  dims.append((k,p,m,q,l))

print(",\n".join([str(el) for el in dims]))
raise Exception(",\n".join([str(el) for el in dims]))
"""

dims = [
(32, 32, 32, 32, 32),
(3, 11, 5, 7, 13),
(10, 19, 87, 59, 94),
(11, 25, 50, 14, 22),
(8, 80, 29, 58, 17),
(13, 19, 11, 100, 48),
(10, 35, 90, 69, 92),
(22, 18, 10, 10, 31),
(14, 40, 46, 20, 17),
(9, 84, 40, 41, 95),
(41, 16, 12, 40, 84),
(19, 25, 30, 9, 41),
(34, 18, 20, 41, 57),
(17, 35, 33, 70, 23),
(15, 16, 40, 100, 73),
(10, 48, 36, 75, 100),
(15, 50, 45, 65, 66),
(10, 95, 87, 81, 45),
(11, 20, 55, 45, 39),
(13, 31, 73, 27, 33),
(9, 98, 67, 76, 34),
(8, 83, 28, 67, 93),
(8, 64, 73, 37, 12),
(20, 40, 33, 9, 49),
(20, 51, 16, 20, 34),
(9, 100, 26, 13, 50),
(29, 24, 30, 34, 30),
(8, 45, 95, 43, 20),
(27, 37, 36, 35, 61),
(9, 48, 52, 75, 27),
(11, 75, 58, 61, 36),
(8, 58, 91, 42, 25),
(9, 15, 61, 33, 67),
(63, 10, 16, 22, 44),
(19, 43, 35, 68, 47),
(9, 97, 83, 56, 15),
(8, 88, 46, 30, 15),
(8, 80, 39, 95, 100),
(20, 11, 21, 41, 84),
(9, 54, 60, 10, 100),
(9, 85, 44, 99, 52),
(29, 13, 28, 84, 79),
(9, 37, 79, 98, 15),
(9, 98, 67, 65, 88),
(34, 21, 13, 39, 79),
(9, 88, 57, 78, 72),
(9, 86, 23, 19, 52),
(12, 63, 85, 53, 8),
(15, 13, 11, 20, 29),
(34, 29, 15, 23, 60)
]

open_bracket = "{"
close_bracket = "}"


for dimId, (K,P,M,Q,L) in enumerate(dims):
  #M*K < 1024
  #M + K*P < 48*1024/4

  #K = 4
  #P = 8
  #M = 12
  #Q = 16
  #L = 20
  shapeA = (K, P, M)
  shapeB = (M, )
  shapeC = (K, Q)
  shapeD = (Q, P)
  shapeE = (K, P, L)
  shapeF = (L, M)
  shapeX = (K, P)
  A = Tensor('A', shapeA)
  B = Tensor('B', shapeB)
  C = Tensor('C', shapeC)
  D = Tensor('D', shapeD)
  E = Tensor('E', shapeE)
  F = Tensor('F', shapeF)
  X = Tensor('X', shapeX)
  a_vector_init = "'k', 'p', 'm', 'b'"
  b_vector_init = "'m', 'b'"
  c_vector_init = "'k', 'q', 'b'"
  d_vector_init = "'q', 'p', 'b'"
  e_vector_init = "'k', 'p', 'l', 'b'"
  f_vector_init = "'l', 'm', 'b'"
  x_vector_init = "'k', 'p', 'b'"


  kernel1 = X['kp'] <= C['kq']  * D['qp']
  kernel2 = A['kpm'] <= A['kpm'] + F['lm'] * E['kpl']
  kernel3 = A['kpm'] <= A['kpm'] + B['m']  * X['kp']


  sizeA = reduce(operator.mul, shapeA, 1)
  sizeB = reduce(operator.mul, shapeB, 1)
  sizeC = reduce(operator.mul, shapeC, 1)
  sizeD = reduce(operator.mul, shapeD, 1)
  sizeE = reduce(operator.mul, shapeE, 1)
  sizeF = reduce(operator.mul, shapeF, 1)
  sizeX = reduce(operator.mul, shapeX, 1)
  maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC), 
                    len(shapeD), len(shapeE), len(shapeF), len(shapeX))

  def get_available_mem_on_gpu():
      meminfo = cuda.current_context().get_memory_info()
      return meminfo[0]


  def get_suggested_num_elements():
      # 1 pointer extra needed per element
      per_el_size = (sizeA + sizeB + sizeC + sizeD + sizeE + sizeF + sizeX) * 4 + (7 * 4)

      available_mem = get_available_mem_on_gpu()
      can_fit_els = available_mem // per_el_size
      lower = int(0.90 * can_fit_els)
      return lower


  num_els = get_suggested_num_elements()

  # Taken from addLocal viscoelastic2.py
  # kernel0 = A['kpm'] <= A['kpm'] + B['m'] * C['kq'] * D['qp'] + E['kpl'] * F['lm']


  gpu_kernels = list()
  function_names = list()
  function_argss = list()

  peakFLOPGiven = tensorPeakFLOP
  peakBandwidthGiven = tensorPeakBandwidth 

  ls_per_el = sizeA*2 + sizeB + sizeC + sizeD + sizeE + sizeF
  ls_per_el *= 4
  ls_unfused_per_el = sizeX + sizeC + sizeD \
                    + sizeA*2 + sizeF + sizeE \
                    + sizeA*2 + sizeB + sizeX
  ls_unfused_per_el *= 4

  k0 = K*Q*P*2
  k1 = L*M*K*P*2 + K*P*M
  k2 = K*M*P*2 + K*M*P
  fp_per_el = k0 + k1 + k2
  fp_unfused_per_el = fp_per_el
  fp_per_k1 = k0
  fp_per_k2 = k1
  fp_per_k3 = k2
  ls_per_k1 = (sizeX   + sizeC + sizeD) * 4
  ls_per_k2 = (sizeA*2 + sizeF + sizeE) * 4
  ls_per_k3 = (sizeA*2 + sizeB + sizeX) * 4

  for kernel in [kernel1, kernel2, kernel3]:
  #for kernel in [kernel1, kernel2, kernel3]:
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
        if line.startswith("void sloopOverGEMM") or line.startswith("void sproduct"):
            fun_split = line.split("(")
            function_name = fun_split[0].split("void ")[1]
            function_args = fun_split[1].split(")")[0]
        if line.startswith("void scopyAddScale"):
          continue

    gpu_kernel = filtered_kernel

    print(function_name)
    print(function_args)

    function_names.append(function_name)
    function_argss.append(function_args)
    gpu_kernels.append(gpu_kernel)

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
    std::cout << __FILE__ << " " << __LINE__ << std::endl;                      \\
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
    }}
    PrevFile = File;
    PrevLine = Line;
#endif
}}

{gpu_kernels[0]}
{gpu_kernels[1]}
{gpu_kernels[2]}

int main(){{
  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els]{{0.f}};
  float* B = new float[{sizeB} * num_els]{{0.f}};
  float* C = new float[{sizeC} * num_els]{{0.f}};
  float* D = new float[{sizeD} * num_els]{{0.f}};
  float* E = new float[{sizeE} * num_els]{{0.f}};
  float* F = new float[{sizeF} * num_els]{{0.f}};
  float* X = new float[{sizeX} * num_els]{{0.f}};
  float* R1 = new float[{sizeA} * num_els]{{0.f}};
  float* R2 = new float[{sizeA} * num_els]{{0.f}};
  //float* Ri1 = new float[{sizeX} * num_els]{{0.f}};
  //float* Ri2 = new float[{sizeA} * num_els]{{0.f}};
  //float* Ri1c = new float[{sizeX} * num_els]{{0.f}};
  //float* Ri2c = new float[{sizeA} * num_els]{{0.f}};


  float* coreA = new float[{sizeA}];
  float* coreB = new float[{sizeB}];
  float* coreC = new float[{sizeC}];
  float* coreD = new float[{sizeD}];
  float* coreE = new float[{sizeE}];
  float* coreF = new float[{sizeF}];

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
  float* X_dev = nullptr;

  float** A_dev_begins = new float*[num_els];
  float** B_dev_begins = new float*[num_els];
  float** C_dev_begins = new float*[num_els];
  float** D_dev_begins = new float*[num_els];
  float** E_dev_begins = new float*[num_els];
  float** F_dev_begins = new float*[num_els];
  float** X_dev_begins = new float*[num_els];

  float** A_dev_begins_dev = nullptr;
  float** B_dev_begins_dev = nullptr;
  float** C_dev_begins_dev = nullptr;
  float** D_dev_begins_dev = nullptr;
  float** E_dev_begins_dev = nullptr;
  float** F_dev_begins_dev = nullptr;
  float** X_dev_begins_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * {sizeD} * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * {sizeE} * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * {sizeF} * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * {sizeX} * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
 
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {sizeB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * {sizeD} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * {sizeE} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * {sizeF} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * {sizeX} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{
    A_dev_begins[i] = A_dev + i * {sizeA};
    B_dev_begins[i] = B_dev + i * {sizeB};
    C_dev_begins[i] = C_dev + i * {sizeC};
    D_dev_begins[i] = D_dev + i * {sizeD};
    E_dev_begins[i] = E_dev + i * {sizeE};
    F_dev_begins[i] = F_dev + i * {sizeF};
    X_dev_begins[i] = X_dev + i * {sizeX};
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {function_names[0]}(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * {sizeX} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Dimensions: " << {K} << ", " << {L} << ", " << {M} << ", " << {P} << ", " << {Q} << ", " << {L} << std::endl;

  float elapsedTimeT1 = 0.0;
  float elapsedTimeT2 = 0.0;
  float elapsedTimeT3 = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEvent_t startT2, stopT2;
  cudaEvent_t startT3, stopT3;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  {function_names[0]}(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT1, startT1, stopT1); CHECK_ERR;
  //cudaDeviceSynchronize(); CHECK_ERR;

  //cudaMemcpy(Ri1, X_dev, sizeof(float) * {sizeX} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventRecord(startT2); CHECK_ERR;
  {function_names[1]}(A_dev_begins_dev, 0, E_dev_begins_dev, 0, F_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT2); CHECK_ERR;
  cudaEventSynchronize(stopT2); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;
  //cudaDeviceSynchronize(); CHECK_ERR;

  //cudaMemcpy(Ri2, A_dev, sizeof(float) * {sizeA} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

  cudaEventCreate(&startT3); CHECK_ERR;
  cudaEventCreate(&stopT3); CHECK_ERR;
  cudaEventRecord(startT3); CHECK_ERR;
  {function_names[2]}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT3); CHECK_ERR;
  cudaEventSynchronize(stopT3); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3); CHECK_ERR;
  double elapsedTime = elapsedTimeT1 + elapsedTimeT2 + elapsedTimeT3;
  cudaDeviceSynchronize(); CHECK_ERR;
  
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, A_dev, sizeof(float) * {sizeA} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  double fp_per_el = {fp_per_el};
  double ls_per_el = {ls_per_el};
  double fp_unfused_per_el = {fp_unfused_per_el};
  double ls_unfused_per_el = {ls_unfused_per_el};
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  fp_unfused_per_el *= num_els;
  ls_unfused_per_el *= num_els;
  std::cout << "Gemmforge Theoretical Fused Kernel GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational Theoretical Fused intensity: " << fp_per_el / ls_per_el << std::endl;
  std::cout << "Gemmforge GFLOPs/s: " << fp_unfused_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational intensity: " << fp_unfused_per_el / ls_unfused_per_el << std::endl;
  double peakFLOPGiven = {peakFLOPGiven};
  double peakBandwidthGiven = {peakBandwidthGiven};

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge" << std::endl;
    //std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;
    double obtainable_unfused_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_unfused_per_el) / static_cast<double>(ls_unfused_per_el)));
    std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTime) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with Gemmforge" << std::endl;
    //std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTime) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with cuTensor" << std::endl;
    double obtainable_unfused_peak_k1 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>({fp_per_k1}) / static_cast<double>({ls_per_k1})));
    std::cout << 100.0*({fp_per_k1} * num_els  * 1e-6 / elapsedTimeT1) / obtainable_unfused_peak_k1 << " % of roof w. respect to Kernel1 intensity achieved with Gemmforge" << std::endl;
    double obtainable_unfused_peak_k2 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>({fp_per_k2}) / static_cast<double>({ls_per_k2})));
    std::cout << 100.0*({fp_per_k2} * num_els  * 1e-6 / elapsedTimeT2) / obtainable_unfused_peak_k2 << " % of roof w. respect to Kernel2 intensity achieved with Gemmforge" << std::endl;
    double obtainable_unfused_peak_k3 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>({fp_per_k3}) / static_cast<double>({ls_per_k3})));
    std::cout << 100.0*({fp_per_k3} * num_els * 1e-6 / elapsedTimeT3) / obtainable_unfused_peak_k3 << " % of roof w. respect to Kernel3 intensity achieved with Gemmforge" << std::endl;
  }}

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * {sizeX} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  cutensorHandle_t* handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cudaEvent_t startCT1, stopCT1;
  cudaEvent_t startCT2, stopCT2;
  cudaEvent_t startCT3, stopCT3;
  cudaEventCreate(&startCT1); CHECK_ERR;
  cudaEventCreate(&stopCT1); CHECK_ERR;
  cudaEventCreate(&startCT2); CHECK_ERR;
  cudaEventCreate(&stopCT2); CHECK_ERR;
  cudaEventCreate(&startCT3); CHECK_ERR;
  cudaEventCreate(&stopCT3); CHECK_ERR;
  float elapsedTimeCT1 = 0.f;
  float elapsedTimeCT2 = 0.f;
  float elapsedTimeCT3 = 0.f;

  // Kernel 1
  std::cout << "cuTensor Kernel 1" << std::endl;
  {{
    float alphaK1 = 1.0f;
    float betaK1 = 0.0f;
    float alphaK2 = 1.0f;
    float betaK2 = 1.0;
    float alphaK3 = 1.0f;
    float betaK3 = 1.0;

    std::vector<int> modeA{open_bracket + a_vector_init + close_bracket};
    std::vector<int> modeB{open_bracket + b_vector_init + close_bracket};
    std::vector<int> modeC{open_bracket + c_vector_init + close_bracket};
    std::vector<int> modeD{open_bracket + d_vector_init + close_bracket};
    std::vector<int> modeE{open_bracket + e_vector_init + close_bracket};
    std::vector<int> modeF{open_bracket + f_vector_init + close_bracket};
    std::vector<int> modeX{open_bracket + x_vector_init + close_bracket};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();
    int nmodeD = modeD.size();
    int nmodeE = modeE.size();
    int nmodeF = modeF.size();
    int nmodeX = modeX.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['k'] = {K};
    extent['l'] = {L};
    extent['m'] = {M};
    extent['p'] = {P};
    extent['q'] = {Q};
    extent['b'] = num_els;

    std::vector<int64_t> extentA;
    for (auto mode : modeA) {{
        extentA.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentB;
    for (auto mode : modeB) {{
        extentB.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentC;
    for (auto mode : modeC) {{
        extentC.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentD;
    for (auto mode : modeD) {{
        extentD.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentE;
    for (auto mode : modeE) {{
        extentE.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentF;
    for (auto mode : modeF) {{
        extentF.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentX;
    for (auto mode : modeX) {{
        extentX.push_back(extent[mode]);
    }}
    
    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeD = CUDA_R_32F;
    cudaDataType_t typeE = CUDA_R_32F;
    cudaDataType_t typeF = CUDA_R_32F;
    cudaDataType_t typeX = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descA,
                    nmodeA,
                    extentA.data(),
                    NULL,
                    typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descB,
                    nmodeB,
                    extentB.data(),
                    NULL,
                    typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor( handle,
                    &descC,
                    nmodeC,
                    extentC.data(),
                    NULL,
                    typeC, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descD;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descD,
                    nmodeD,
                    extentD.data(),
                    NULL,
                    typeD, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descE;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descE,
                    nmodeE,
                    extentE.data(),
                    NULL,
                    typeE, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descF;
    HANDLE_ERROR(cutensorInitTensorDescriptor( handle,
                    &descF,
                    nmodeF,
                    extentF.data(),
                    NULL,
                    typeF, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descX;
    HANDLE_ERROR(cutensorInitTensorDescriptor( handle,
                    &descX,
                    nmodeX,
                    extentX.data(),
                    NULL,
                    typeX, CUTENSOR_OP_IDENTITY));


    uint32_t alignmentRequirementA;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    A_dev,
                    &descA,
                    &alignmentRequirementA));

    uint32_t alignmentRequirementB;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    B_dev,
                    &descB,
                    &alignmentRequirementB));

    uint32_t alignmentRequirementC;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    C_dev,
                    &descC, 
                    &alignmentRequirementC));

    uint32_t alignmentRequirementD;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    D_dev,
                    &descD,
                    &alignmentRequirementD));

    uint32_t alignmentRequirementE;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    E_dev,
                    &descE,
                    &alignmentRequirementE));

    uint32_t alignmentRequirementF;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    F_dev,
                    &descF, 
                    &alignmentRequirementF));

    uint32_t alignmentRequirementX;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    X_dev,
                    &descX, 
                    &alignmentRequirementX));

    cutensorContractionDescriptor_t desc1;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc1,
                  &descC, modeC.data(), alignmentRequirementC,
                  &descD, modeD.data(), alignmentRequirementD,
                  &descX, modeX.data(), alignmentRequirementX,
                  &descX, modeX.data(), alignmentRequirementX,
                  typeCompute));

    cutensorContractionFind_t find1;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 handle, &find1, 
                 CUTENSOR_ALGO_DEFAULT));

    uint64_t worksize1 = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle,
                 &desc1,
                 &find1,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize1));

    cutensorContractionDescriptor_t desc2;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc2,
                  &descF, modeF.data(), alignmentRequirementF,
                  &descE, modeE.data(), alignmentRequirementE,
                  &descA, modeA.data(), alignmentRequirementA,
                  &descA, modeA.data(), alignmentRequirementA,
                  typeCompute));

    cutensorContractionFind_t find2;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 handle, &find2, 
                 CUTENSOR_ALGO_DEFAULT));

    uint64_t worksize2 = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle,
                 &desc2,
                 &find2,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize2));


    cutensorContractionDescriptor_t desc3;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc3,
                  &descB, modeB.data(), alignmentRequirementB,
                  &descX, modeX.data(), alignmentRequirementX,
                  &descA, modeA.data(), alignmentRequirementA,
                  &descA, modeA.data(), alignmentRequirementA,
                  typeCompute));

    cutensorContractionFind_t find3;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 handle, &find3, 
                 CUTENSOR_ALGO_DEFAULT));

    uint64_t worksize3 = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle,
                 &desc3,
                 &find3,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize3));

    uint64_t maxWorkSize = std::max(std::max(worksize1, worksize2), worksize3);
    void *work = nullptr;
    if (maxWorkSize > 0)
    {{
        if (cudaSuccess != cudaMalloc(&work, maxWorkSize))
        {{
            work = nullptr;
            maxWorkSize = 0;
            worksize1 = 0;
            worksize2 = 0;
            worksize3 = 0;
            cudaGetLastError(); // Clear last error to save CHECK_ERR;
        }} else {{
            worksize1 = maxWorkSize;
            worksize2 = maxWorkSize;
            worksize3 = maxWorkSize;
        }}
    }}


    cutensorContractionPlan_t plan1;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan1,
                 &desc1,
                 &find1,
                 worksize1));

    cutensorContractionPlan_t plan2;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan2,
                 &desc2,
                 &find2,
                 worksize2));

    cutensorContractionPlan_t plan3;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan3,
                 &desc3,
                 &find3,
                 worksize3));

    cudaDeviceSynchronize(); CHECK_ERR;

    cudaEventRecord(startCT1); CHECK_ERR;
    cutensorContraction(handle,
                              &plan1,
                              (void*) &alphaK1, C_dev, D_dev,
                              (void*) &betaK1,  X_dev, X_dev, 
                              work, worksize1, 0);
    cudaEventRecord(stopCT1); CHECK_ERR;
    cudaEventSynchronize(stopCT1); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTimeCT1, startCT1, stopCT1); CHECK_ERR;

    //cudaDeviceSynchronize(); CHECK_ERR;
    //cudaMemcpy(Ri1c, X_dev, sizeof(float) * {sizeX} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

    cudaEventRecord(startCT2); CHECK_ERR;
    cutensorContraction(handle,
                              &plan2,
                              (void*) &alphaK2, F_dev, E_dev,
                              (void*) &betaK2,  A_dev, A_dev, 
                              work, worksize2, 0);
    cudaEventRecord(stopCT2); CHECK_ERR;
    cudaEventSynchronize(stopCT2); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTimeCT2, startCT2, stopCT2); CHECK_ERR;

    //cudaDeviceSynchronize(); CHECK_ERR;
    //cudaMemcpy(Ri2c, A_dev, sizeof(float) * {sizeA} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

    cudaEventRecord(startCT3); CHECK_ERR;
    cutensorContraction(handle,
                              &plan3,
                              (void*) &alphaK3, B_dev, X_dev,
                              (void*) &betaK3,  A_dev, A_dev, 
                              work, worksize3, 0);
    cudaEventRecord(stopCT3); CHECK_ERR;
    cudaEventSynchronize(stopCT3); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTimeCT3, startCT3, stopCT3); CHECK_ERR;

    cudaDeviceSynchronize(); CHECK_ERR;
    
    cudaMemcpy(R2, A_dev, sizeof(float) * {sizeA} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

    cudaFree(work);
  }}

  float elapsedTimeCuTensor = elapsedTimeCT1 + elapsedTimeCT2 + elapsedTimeCT2;
  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTimeCuTensor) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;

    double obtainable_unfused_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_unfused_per_el) / static_cast<double>(ls_unfused_per_el)));
    std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTimeCuTensor) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with cuTensor" << std::endl;
  }}

  /*
  bool i1results_wrong = false;
  for (size_t i = 0; i < {sizeX} * num_els; i++){{
    if (std::abs(Ri1[i] - Ri1c[i]) > 1.0f) {{
      std::cout << "Intermediate Results 1 do not match, problem first at offset " << i << " :_(" << std::endl;
      i1results_wrong = true;
      break;
    }}
  }}
  if (!i1results_wrong){{
    std::cout << "Gemmforge and cuTensor contraction intermediate results 1 match! :)" << std::endl;
  }}
  
  bool i2results_wrong = false;
  for (size_t i = 0; i < {sizeA} * num_els; i++){{
    if (std::abs(Ri2[i] - Ri2c[i]) > 1.0f) {{
      std::cout << "Intermediate Results 2 do not match, problem first at offset " << i << " :_(" << std::endl;
      i2results_wrong = true;
      break;
    }}
  }}
  if (!i2results_wrong){{
    std::cout << "Gemmforge and cuTensor contraction intermediate results 2 match! :)" << std::endl;
  }}
  */

  bool results_wrong = false;
  for (size_t i = 0; i < {sizeA} * num_els; i++){{
    if (std::abs(R1[i] - R2[i]) > 5.0f) {{
      std::cout << "Results do not match, problem first at offset " << i << " :_(" << std::endl;
      results_wrong = true;
      break;
    }}
  }}
  if (!results_wrong){{
    std::cout << "Gemmforge and cuTensor contraction results match! :)" << std::endl;
  }}

  cudaFree(A_dev_begins_dev);
  cudaFree(B_dev_begins_dev);
  cudaFree(C_dev_begins_dev);
  cudaFree(D_dev_begins_dev);
  cudaFree(E_dev_begins_dev);
  cudaFree(F_dev_begins_dev);
  cudaFree(X_dev_begins_dev);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  delete[] E;
  delete[] F;
  delete[] X;
  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;
  delete[] D_dev_begins;
  delete[] E_dev_begins;
  delete[] F_dev_begins;
  delete[] X_dev_begins;
  delete[] R1;
  delete[] R2;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  cudaFree(D_dev);
  cudaFree(E_dev);
  cudaFree(F_dev);
  cudaFree(X_dev);

  delete[] coreA;
  delete[] coreB;
  delete[] coreC;
  delete[] coreD;
  delete[] coreE;
  delete[] coreF;

  return 0;
}}

"""

  if dimId < 10:
    code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_2_variant_0{dimId}.cu", "w")
  else:
    code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_2_variant_{dimId}.cu", "w")
  code_file.write(benchmark_str)
  code_file.flush()
  code_file.close()
