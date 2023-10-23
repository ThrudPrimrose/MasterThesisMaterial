import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda
import random
import itertools

dims = [
  (16, 16, 16, 16, 16)
]

open_bracket = "{"
close_bracket = "}"

N = 16


pA = ["'i'", "'k'", "'x'"]
pB = ["'x'", "'j'", "'k'"]
pC = ["'i'", "'j'", "'k'"]
dimId = 0
for p1 in list(itertools.permutations(pA)):
  for p2 in list(itertools.permutations(pB)):
    dimId +=1
    print (p1, p2)
    shapeA = (N, N, N)
    shapeB = (N, N, N)
    shapeC = (N, N, N)
    A = Tensor('A', shapeA)
    B = Tensor('B', shapeB)
    C = Tensor('C', shapeC)
    a_vector_init = ", ".join(pA) + ", 'b'"
    b_vector_init = ", ".join(pB) + ", 'b'"
    c_vector_init = ", ".join(pC) + ", 'b'"

    yA = "".join(pA).replace("'", "")
    yB = "".join(pB).replace("'", "")
    yC = "".join(pC).replace("'", "")

    kernel1 = C[yC] <= A[yA]  * B[yB]


    sizeA = reduce(operator.mul, shapeA, 1)
    sizeB = reduce(operator.mul, shapeB, 1)
    sizeC = reduce(operator.mul, shapeC, 1)
    maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC))

    def get_available_mem_on_gpu():
        meminfo = cuda.current_context().get_memory_info()
        return meminfo[0]


    def get_suggested_num_elements():
        # 1 pointer extra needed per element
        per_el_size = (sizeA + sizeB + sizeC) * 4 + (3 * 4)

        available_mem = get_available_mem_on_gpu()
        can_fit_els = available_mem // per_el_size
        lower = int(0.90 * can_fit_els)
        return lower


    num_els = get_suggested_num_elements()

    gpu_kernels = list()
    function_names = list()
    function_argss = list()

    peakFLOPGiven = tensorPeakFLOP
    peakBandwidthGiven = tensorPeakBandwidth 

    ls_per_el = sizeC*2 + sizeA + sizeB
    ls_per_el *= 4
    ls_unfused_per_el = sizeC*2 \
                      + sizeA \
                      + sizeB
    ls_unfused_per_el *= 4

    k0 = N*N*N*2
    fp_per_el = k0
    fp_unfused_per_el = fp_per_el


    for kernel in [kernel1]:
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

int main(){{
  std::cout << "Kernel {dimId}" << std::endl;
  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els]{{0.f}};
  float* B = new float[{sizeB} * num_els]{{0.f}};
  float* C = new float[{sizeC} * num_els]{{0.f}};
  float* R1 = new float[{sizeC} * num_els]{{0.f}};
  float* R2 = new float[{sizeC} * num_els]{{0.f}};

  float* coreA = new float[{sizeA}];
  float* coreB = new float[{sizeB}];
  float* coreC = new float[{sizeC}];

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

  for (size_t i = 0; i < num_els; i++){{
      std::memcpy(&A[i * {sizeA}], &coreA[0], {sizeA} * sizeof(float));
      std::memcpy(&B[i * {sizeB}], &coreB[0], {sizeB} * sizeof(float));
      std::memcpy(&C[i * {sizeC}], &coreC[0], {sizeC} * sizeof(float));
  }}

  float* A_dev = nullptr;
  float* B_dev = nullptr;
  float* C_dev = nullptr;

  float** A_dev_begins = new float*[num_els];
  float** B_dev_begins = new float*[num_els];
  float** C_dev_begins = new float*[num_els];

  float** A_dev_begins_dev = nullptr;
  float** B_dev_begins_dev = nullptr;
  float** C_dev_begins_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;

  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {sizeB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{
    A_dev_begins[i] = A_dev + i * {sizeA};
    B_dev_begins[i] = B_dev + i * {sizeB};
    C_dev_begins[i] = C_dev + i * {sizeC};
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {function_names[0]}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Dimensions: " << {N} << ", " << {N} << ", " << {N} << std::endl;

  float elapsedTimeT1 = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  {function_names[0]}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT1, startT1, stopT1); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTimeT1 << " ms" << std::endl; 
  cudaMemcpy(R1, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  double fp_per_el = {fp_per_el};
  double ls_per_el = {ls_per_el};
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  std::cout << "Gemmforge Kernel GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTimeT1 << std::endl;
  std::cout << "Operational intensity: " << fp_per_el / ls_per_el << std::endl;
  double peakFLOPGiven = {peakFLOPGiven};
  double peakBandwidthGiven = {peakBandwidthGiven};

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTimeT1) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge" << std::endl;
  }}

  cutensorHandle_t* handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cudaEvent_t startCT1, stopCT1;
  cudaEventCreate(&startCT1); CHECK_ERR;
  cudaEventCreate(&stopCT1); CHECK_ERR;
  float elapsedTimeCT1 = 0.f;

  // Kernel 1
  std::cout << "cuTensor Kernel 1" << std::endl;
  {{
    float alphaK1 = 1.0f;
    float betaK1 = 0.0f;

    std::vector<int> modeA{open_bracket + a_vector_init + close_bracket};
    std::vector<int> modeB{open_bracket + b_vector_init + close_bracket};
    std::vector<int> modeC{open_bracket + c_vector_init + close_bracket};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = {N};
    extent['j'] = {N};
    extent['k'] = {N};
    extent['x'] = {N};
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


    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
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

    cutensorContractionDescriptor_t desc1;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc1,
                  &descA, modeA.data(), alignmentRequirementA,
                  &descB, modeB.data(), alignmentRequirementB,
                  &descC, modeC.data(), alignmentRequirementC,
                  &descC, modeC.data(), alignmentRequirementC,
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

    uint64_t maxWorkSize = worksize1;
    void *work = nullptr;
    if (maxWorkSize > 0)
    {{
        if (cudaSuccess != cudaMalloc(&work, maxWorkSize))
        {{
            work = nullptr;
            maxWorkSize = 0;
            worksize1 = 0;
            cudaGetLastError(); // Clear last error to save CHECK_ERR;
        }} else {{
            worksize1 = maxWorkSize;
        }}
    }}

    cutensorContractionPlan_t plan1;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan1,
                 &desc1,
                 &find1,
                 worksize1));

    cudaDeviceSynchronize(); CHECK_ERR;

    cudaEventRecord(startCT1); CHECK_ERR;
    cutensorContraction(handle,
                              &plan1,
                              (void*) &alphaK1, A_dev, B_dev,
                              (void*) &betaK1,  C_dev, C_dev, 
                              work, worksize1, 0);
    cudaEventRecord(stopCT1); CHECK_ERR;
    cudaEventSynchronize(stopCT1); CHECK_ERR;
    cudaEventElapsedTime(&elapsedTimeCT1, startCT1, stopCT1); CHECK_ERR;

    cudaDeviceSynchronize(); CHECK_ERR;


    cudaMemcpy(R2, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

    cudaFree(work);
  }}

  float elapsedTimeCuTensor = elapsedTimeCT1;
  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTimeCuTensor) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;
  }}

  bool results_wrong = false;
  for (size_t i = 0; i < {sizeC} * num_els; i++){{
    if (std::abs(R1[i] - R2[i]) > 1.0f) {{
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

  delete[] A;
  delete[] B;
  delete[] C;

  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;

  delete[] R1;
  delete[] R2;

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  delete[] coreA;
  delete[] coreB;
  delete[] coreC;


  return 0;
}}

"""

      if dimId < 10:
        code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_3_variant_0{dimId}.cu", "w")
      else:
        code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_3_variant_{dimId}.cu", "w")
      code_file.write(benchmark_str)
      code_file.flush()
      code_file.close()
