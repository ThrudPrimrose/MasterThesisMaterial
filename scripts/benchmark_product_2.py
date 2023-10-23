import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda
import random

data_dir = f"/tmp"

"""
shrmem_limit = 48*1024
ijk_s = list()
while len(ijk_s) < 14:
  random_integer_i = random.randint(1, 24)
  random_integer_j = random.randint(1, 24)
  random_integer_k = random.randint(2, 24)
  element_count = random_integer_i * random_integer_j * random_integer_k
  if element_count * 8 > shrmem_limit:
    continue
  if element_count // random_integer_i > 1024:
    continue
  ijk_s.append((random_integer_i, random_integer_j, random_integer_k))
"""

#Tthe following dimensions were generated:
ijk_s = [(21, 4, 23), (12, 9, 23), (11, 10, 12), 
          (14, 10, 12), (7, 8, 10), (7, 5, 14), 
          (22, 12, 3), (17, 5, 11), (18, 10, 20), 
          (18, 6, 6), (19, 19, 17), (16, 6, 5), 
          (14, 14, 4), (6, 4, 13)]

for v, (I, J, K) in enumerate(ijk_s):

  A = Tensor('A', (I, J, K))
  B = Tensor('B', (K, J, I))
  C = Tensor('C', (I, J, K))

  shapeA = (I, J, K)
  shapeB = (K, J, I)
  shapeC = (I, J, K)
  A = Tensor('A', shapeA)
  B = Tensor('B', shapeB)
  C = Tensor('C', shapeC)
  sizeA = reduce(operator.mul, shapeA, 1)
  sizeB = reduce(operator.mul, shapeB, 1)
  sizeC = reduce(operator.mul, shapeC, 1)
  maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC))


  a_tuple = ",".join(str(item) for item in shapeA)
  b_tuple = ",".join(str(item) for item in shapeB)
  c_tuple = ",".join(str(item) for item in shapeC)
  shape_str = ""
  shape_str += f"A({a_tuple}), B({b_tuple}), C({c_tuple})"


  def get_available_mem_on_gpu():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]


  def get_suggested_num_elements():
    # 1 pointer extra needed per element
    per_el_size = (sizeA + sizeB + sizeC) * 4 + 12

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    lower = int(0.60 * can_fit_els)
    return lower

  FLOAT_SIZE = 4

  num_els = get_suggested_num_elements()

  """
  fp_ld_per_els = [((N*N*N*2, N*N*N*3*FLOAT_SIZE))]
  permutations = [("ijk", "ijk")]
  test_case_count = 1
  """

  fp_ld_per_els = [(I*J*K*2, I*J*K*3*FLOAT_SIZE)]

  permutations = [("ijk", "kji")]

  test_case_count = 1


  peakBandwidthGiven = tensorPeakBandwidth
  peakFLOPGiven = tensorPeakFLOP
  open_bracket ="{"
  close_bracket="}"

  i = 0
  (a_c_perm, b_perm) = permutations[i]
  (fp_per_el, ls_per_el) = fp_ld_per_els[i]

  a_c_vector_init = str()
  b_vector_init = str()
  for c in a_c_perm:
    a_c_vector_init += f"'{c}', "
  for c in b_perm:
    b_vector_init += f"'{c}', "
  a_c_vector_init += "'b'"
  b_vector_init += "'b'"

  kernel = C[a_c_perm] <= A[a_c_perm] * B[b_perm]

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
      if line.startswith("void sproduct"):
          fun_split = line.split("(")
          function_name = fun_split[0].split("void ")[1]
          function_args = fun_split[1].split(")")[0]
      """
      if "__shared__  __align__" in line:
        tokens = line.split("[")
        alloc_size = tokens[1].split("]")[0]
        alloc_base = tokens[0]
        dynamic_alloc = "extern " + alloc_base + "[];"
        #dynamic_alloc = dynamic_alloc.replace("float ", "float* ")
        filtered_kernel += dynamic_alloc + "\n"
        continue
      elif "dim3 grid" in line:
        filtered_kernel += line + "\n"
        filtered_kernel += f"size_t shr_mem_size = sizeof(float)*({sizeA} + {sizeB});\n"
      elif "kernel_sproduct_" in line and "<<<" in line and ">>>" in line:
        nl = line.replace("<<<grid,block,0,stream>>>", "<<<grid,block,shr_mem_size,stream>>>")
        filtered_kernel += nl + "\n"
      else:
      """
      filtered_kernel += line + "\n"

  gpu_kernel = filtered_kernel

  print(function_name)
  print(function_args)
  print(gpu_kernel)

  a = A.memoryLayout
  print(a)


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


int main(){{
  //cudaFuncSetAttribute(kernel_{function_name}, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304); CHECK_ERR;


  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els];
  float* B = new float[{sizeB} * num_els];
  float* C = new float[{sizeC} * num_els];
  float* R1 = new float[{sizeC} * num_els]{{0.f}};
  float* R2 = new float[{sizeC} * num_els]{{0.f}};

  float coreA[{sizeA}];
  float coreB[{sizeB}];
  float coreC[{sizeC}];

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

  {function_name}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Will compute the kernel: C['{a_c_perm}'] <= A['{a_c_perm}'] * B['{b_perm}'], with Gemmforge" << std::endl;
   std::cout << "Shapes and dims: " << "{shape_str}" << std::endl;
  float elapsedTime = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  {function_name}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startT1, stopT1); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  std::cout << "Will compute the kernel: C[{a_c_perm}] <= A[{a_c_perm}] * B[{b_perm}], with cuTensor" << std::endl;

  cutensorHandle_t* handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cudaEvent_t startT2, stopT2;
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;

  cudaFree(A_dev_begins_dev); CHECK_ERR;
  cudaFree(B_dev_begins_dev); CHECK_ERR;
  cudaFree(C_dev_begins_dev); CHECK_ERR;

  // Kernel 1
  std::cout << "cuTensor Kernel 1" << std::endl;
  {{
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<int> modeA{open_bracket + a_c_vector_init + close_bracket};
    std::vector<int> modeB{open_bracket + b_vector_init + close_bracket};
    std::vector<int> modeC{open_bracket + a_c_vector_init + close_bracket};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = {I};
    extent['j'] = {J};
    extent['k'] = {K};
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

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc,
                  &descA, modeA.data(), alignmentRequirementA,
                  &descB, modeB.data(), alignmentRequirementB,
                  &descC, modeC.data(), alignmentRequirementC,
                  &descC, modeC.data(), alignmentRequirementC,
                  typeCompute));

    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 handle, &find, 
                 CUTENSOR_ALGO_DEFAULT));

    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void *work = nullptr;
    if (worksize > 0)
    {{
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {{
            work = nullptr;
            worksize = 0;
            cudaGetLastError(); // Clear last error to save CHECK_ERR;
        }}
    }}

    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    cudaDeviceSynchronize(); CHECK_ERR;

    cutensorStatus_t err;
    err = cutensorContraction(handle,
                              &plan,
                              (void*) &alpha, A_dev, B_dev,
                              (void*) &beta,  C_dev, C_dev, 
                              work, worksize, 0);
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaMemset(C_dev, 0.0f, {I * J * K} * num_els); CHECK_ERR;

    cudaEventRecord(startT2); CHECK_ERR;
    err = cutensorContraction(handle,
                              &plan,
                              (void*) &alpha, A_dev, B_dev,
                              (void*) &beta,  C_dev, C_dev, 
                              work, worksize, 0);
    cudaEventRecord(stopT2); CHECK_ERR;

    cudaDeviceSynchronize(); CHECK_ERR;
    cudaMemcpy(R2, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
    cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

    if (err != CUTENSOR_STATUS_SUCCESS)
    {{
      printf("ERROR: %s in line %d\\n", cutensorGetErrorString(err), __LINE__);
    }}
    else
    {{
      printf("Sub-kernel 1 succeeded.\\n");
    }}

    cudaFree(work);

  }}

  float elapsedTimeT2 = 0.f;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;
  std::cout << "cuTensor Tensor Contraction took: " << elapsedTimeT2 << " ms" << std::endl; 

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

  double fp_per_el = {fp_per_el};
  double ls_per_el = {ls_per_el};
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  std::cout << "Gemmfor GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational intensity: " << fp_per_el / ls_per_el << std::endl;
 
  double peakFLOPGiven = {peakFLOPGiven};
  double peakBandwidthGiven = {peakBandwidthGiven};

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge" << std::endl;
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTimeT2) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;
  }}

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;
  delete[] R1;
  delete[] R2;

  cudaFree(A_dev);
  cudaFree(C_dev);
  cudaFree(B_dev);

  return 0;
}}

"""

  code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_product_2_{a_c_perm}_{b_perm}_{v}.cu", "w")
  code_file.write(benchmark_str)
  code_file.flush()
  code_file.close()

