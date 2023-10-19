import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda


dims = [[(11, 73), 
         (31, 27, 11),
         (27,),
         (31, 73)],
        [(48, 144), 
         (96, 16, 48),
         (16,),
         (96, 144)],
        [(32, 32), 
         (32, 32, 32),
         (32,),
         (32, 32)],
        [(16, 16), 
         (16, 16, 16),
         (16,),
         (16, 16)],
        [(3, 11), 
         (5, 7, 3),
         (7,),
         (5, 11)],
        [(53, 107), 
         (101, 23, 53),
         (23,),
         (101, 107)]]

for dimId in range(6):
  shapeA = dims[dimId][0]
  shapeB = dims[dimId][1]
  shapew = dims[dimId][2]
  shapeC = dims[dimId][3]
  A = Tensor('A', shapeA)
  B = Tensor('B', shapeB)
  w = Tensor('w', shapew)
  C = Tensor('C', shapeC)
  sizeA = reduce(operator.mul, shapeA, 1)
  sizeB = reduce(operator.mul, shapeB, 1)
  sizew = reduce(operator.mul, shapew, 1)
  sizeC = reduce(operator.mul, shapeC, 1)
  maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC), len(shapew))

  a_tuple = ",".join(str(item) for item in shapeA)
  b_tuple = ",".join(str(item) for item in shapeB)
  c_tuple = ",".join(str(item) for item in shapeC)
  w_tuple = ",".join(str(item) for item in shapew)
  shape_str = ""
  shape_str += f"A({a_tuple}), B({b_tuple}), C({c_tuple}), w({w_tuple})"

  kernel = C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k']

  def get_available_mem_on_gpu():
      meminfo = cuda.current_context().get_memory_info()
      return meminfo[0]


  def get_suggested_num_elements():
      # 1 pointer extra needed per element
      per_el_size = (sizeA + sizeB + sizew + sizeC) * 4 + 16

      available_mem = get_available_mem_on_gpu()
      can_fit_els = available_mem // per_el_size
      lower = int(0.70 * can_fit_els)
      return lower

  fp_per_el_0 = 0
  fp_per_el_0 += 32*27*2 # Comp loop first kernel
  fp_per_el_0 += 32*12*64*2 # Comp loop second kernel
  fp_per_el_0 += 32*64 # C += second kernel

  fp_per_el_1 = 0
  fp_per_el_1 += 96*16*2 # Comp loop first kernel
  fp_per_el_1 += 96*48*144*2 # Comp loop second kernel
  fp_per_el_1 += 96*144 # C += second kernel

  fp_per_el_2 = 0
  fp_per_el_2 += 32*32*2 # Comp loop first kernel
  fp_per_el_2 += 32*32*32*2 # Comp loop second kernel
  fp_per_el_2 += 32*32 # C += second kernel

  fp_per_el_3 = 0
  fp_per_el_3 += 16*16*2 # Comp loop first kernel
  fp_per_el_3 += 16*16*16*2 # Comp loop second kernel
  fp_per_el_3 += 16*16 # C += second kernel

  fp_per_el_4 = 0
  fp_per_el_4 += 5*7*2 # Comp loop first kernel
  fp_per_el_4 += 5*3*11*2 # Comp loop second kernel
  fp_per_el_4 += 5*11 # C += second kernel

  fp_per_el_5 = 0
  fp_per_el_5 += 101*23*2 # Comp loop first kernel
  fp_per_el_5 += 101*53*107*2 # Comp loop second kernel
  fp_per_el_5 += 101*107 # C += second kernel

  fp_per_els = [fp_per_el_0, fp_per_el_1, fp_per_el_2, fp_per_el_3, fp_per_el_4, fp_per_el_5]
  fp_per_el = fp_per_els[dimId]

  ls_per_el = 0
  ls_per_el += sizeA # load w first kernel
  ls_per_el += sizeB # load B first kernel
  ls_per_el += sizeC*2 # load A second kernel
  ls_per_el += sizew # load and write C second kernel
  ls_per_el *= 4

  peakBandwidthGiven = tensorPeakBandwidth
  peakFLOPGiven = tensorPeakFLOP

  num_els = get_suggested_num_elements()



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
  size_t currentAllocSize = 0;

  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els];
  float* B = new float[{sizeB} * num_els];
  float* C = new float[{sizeC} * num_els];
  float* w = new float[{sizew} * num_els];
  float* R1 = new float[{sizeC} * num_els]{{0.f}};
  float* R2 = new float[{sizeC} * num_els]{{0.f}};

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

  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev, sizeof(float) * {sizew} * num_els); CHECK_ERR;
  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  currentAllocSize += sizeof(float) * {sizeA} * num_els +
                      sizeof(float) * {sizeB} * num_els +
                      sizeof(float) * {sizeC} * num_els +
                      sizeof(float) * {sizew} * num_els +
                      4 * sizeof(float*) * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

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

  std::cout << "Will compute the kernel: C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k'], with Gemmforge" << std::endl;
  std::cout << "Shapes and dims: " << "{shape_str}" << std::endl;
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
  cudaMemcpy(R1, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  std::cout << "Will compute the kernel: C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k'], with cuTensor" << std::endl;
  std::cout << "Need to split into 2 kernels, 1: X['il'] <= B['ikl'] * w['k'], with cuTensor" << std::endl;
  std::cout <<"Need to split into 2 kernels, 2: C['ij'] <=  A['lj'] * X['il'], with cuTensor" << std::endl;
  std::cout << "Batched version managed through: C['ijb'] <= C['ijb'] + A['ljb'] * B['iklb'] * w['kb'], with cuTensor" << std::endl;

  float* X_dev = nullptr;
  cudaMalloc((void **)&X_dev, sizeof(float) * {shapeB[0]} * {shapeB[2]} * num_els); CHECK_ERR;
  currentAllocSize += sizeof(float) * {shapeB[0]} * {shapeB[2]} * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

  cutensorHandle_t* handle;
  HANDLE_ERROR(cutensorCreate(&handle));

  cudaEvent_t startT2, stopT2;
  cudaEvent_t startT3, stopT3;
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventCreate(&startT3); CHECK_ERR;
  cudaEventCreate(&stopT3); CHECK_ERR;

  cudaFree(A_dev_begins_dev); CHECK_ERR;
  cudaFree(B_dev_begins_dev); CHECK_ERR;
  cudaFree(C_dev_begins_dev); CHECK_ERR;
  cudaFree(w_dev_begins_dev); CHECK_ERR;
  currentAllocSize -= sizeof(float*) * 4 * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;


  // Kernel 1
  std::cout << "cuTensor Kernel 1" << std::endl;
  {{
    float alpha = 1.0f;
    float beta = 0.0f;

    // X['il'] <= B['ikl'] * w['k']
    std::vector<int> modeX{{'i', 'l', 'b'}};
    std::vector<int> modeB{{'i', 'k', 'l', 'b'}};
    std::vector<int> modew{{'k', 'b'}};
    int nmodeX = modeX.size();
    int nmodeB = modeB.size();
    int nmodew = modew.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = {shapeB[0]};
    extent['k'] = {shapeB[1]};
    extent['l'] = {shapeB[2]};
    extent['b'] = num_els;

    std::vector<int64_t> extentX;
    for (auto mode : modeX) {{
        extentX.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentB;
    for (auto mode : modeB) {{
        extentB.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentw;
    for (auto mode : modew) {{
        extentw.push_back(extent[mode]);
    }}

    cudaDataType_t typeX = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typew = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    cutensorTensorDescriptor_t descX;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descX,
                    nmodeX,
                    extentX.data(),
                    NULL,
                    typeX, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descB,
                    nmodeB,
                    extentB.data(),
                    NULL,
                    typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descw;
    HANDLE_ERROR(cutensorInitTensorDescriptor( handle,
                    &descw,
                    nmodew,
                    extentw.data(),
                    NULL,
                    typew, CUTENSOR_OP_IDENTITY));

    uint32_t alignmentRequirementX;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    X_dev,
                    &descX,
                    &alignmentRequirementX));

    uint32_t alignmentRequirementB;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    B_dev,
                    &descB,
                    &alignmentRequirementB));

    uint32_t alignmentRequirementw;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    w_dev,
                    &descw, 
                    &alignmentRequirementw));

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc,
                  &descB, modeB.data(), alignmentRequirementB,
                  &descw, modew.data(), alignmentRequirementw,
                  &descX, modeX.data(), alignmentRequirementX,
                  &descX, modeX.data(), alignmentRequirementX,
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
    currentAllocSize += worksize;
    std::cout << "Alloc additional buffer: " << static_cast<float>(worksize) / (1024.0 * 1024.0 * 1024.0) << std::endl;
    std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

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
                              (void*) &alpha, B_dev, w_dev,
                              (void*) &beta,  X_dev, X_dev, 
                              work, worksize, 0);
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaMemset(X_dev, 0.0f, {shapeB[0] * shapeB[1]} * num_els); CHECK_ERR;

    cudaEventRecord(startT2); CHECK_ERR;
    err = cutensorContraction(handle,
                              &plan,
                              (void*) &alpha, B_dev, w_dev,
                              (void*) &beta,  X_dev, X_dev, 
                              work, worksize, 0);
    cudaEventRecord(stopT2); CHECK_ERR;

    cudaDeviceSynchronize(); CHECK_ERR;

    if (err != CUTENSOR_STATUS_SUCCESS)
    {{
      printf("ERROR: %s in line %d\\n", cutensorGetErrorString(err), __LINE__);
    }}
    else
    {{
      printf("Sub-kernel 1 succeeded.\\n");
    }}

    cudaFree(work);
    currentAllocSize -= worksize;
    std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

  }}

  float elapsedTimeT2 = 0.0f;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;
  std::cout << "cuTensor sub-kernel 1 took: " << elapsedTimeT2 << " ms" << std::endl;

  // Kernel 2
  std::cout << "cuTensor Kernel 2" << std::endl;
  {{
    float alpha = 1.0f;
    float beta = 1.0f;
    // C['ij'] <=  C['ij'] + A['lj'] * X['il']
    std::vector<int> modeA{{'l', 'j', 'b'}};
    std::vector<int> modeC{{'i', 'j', 'b'}};
    std::vector<int> modeX{{'i', 'l', 'b'}};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();
    int nmodeX = modeX.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = {shapeC[0]};
    extent['j'] = {shapeC[1]};
    extent['l'] = {shapeA[0]};
    extent['b'] = num_els;

    std::vector<int64_t> extentA;
    for (auto mode : modeA) {{
        extentA.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentC;
    for (auto mode : modeC) {{
        extentC.push_back(extent[mode]);
    }}
    std::vector<int64_t> extentX;
    for (auto mode : modeX) {{
        extentX.push_back(extent[mode]);
    }}

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeX = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descA,
                    nmodeA,
                    extentA.data(),
                    NULL,
                    typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor(handle,
                    &descC,
                    nmodeC,
                    extentC.data(),
                    NULL,
                    typeC, CUTENSOR_OP_IDENTITY));

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

    uint32_t alignmentRequirementC;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    C_dev,
                    &descC,
                    &alignmentRequirementC));

    uint32_t alignmentRequirementX;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(handle,
                    X_dev,
                    &descX, 
                    &alignmentRequirementX));

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(handle, 
                  &desc,
                  &descA, modeA.data(), alignmentRequirementA,
                  &descX, modeX.data(), alignmentRequirementX,
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
    currentAllocSize += worksize;
    std::cout << "Alloc additional buffer: " << static_cast<float>(worksize) / (1024.0 * 1024.0 * 1024.0) << std::endl;
    std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;


    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    cutensorStatus_t err;
    err = cutensorContraction(handle,
                              &plan,
                              (void*) &alpha, A_dev, X_dev,
                              (void*) &beta,  C_dev, C_dev, 
                              work, worksize, 0);
    cudaDeviceSynchronize(); CHECK_ERR;
    cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

    cudaEventRecord(startT3); CHECK_ERR;
    err = cutensorContraction(handle,
                              &plan,
                              (void*) &alpha, A_dev, X_dev,
                              (void*) &beta,  C_dev, C_dev, 
                              work, worksize, 0);
    cudaEventRecord(stopT3); CHECK_ERR;

    cudaDeviceSynchronize();

    if (err != CUTENSOR_STATUS_SUCCESS)
    {{
      printf("ERROR: %s in line %d\\n", cutensorGetErrorString(err), __LINE__);
    }}
    else
    {{
      printf("Sub-kernel 1 succeeded.\\n");
    }}

    cudaFree(work);
    currentAllocSize -= worksize;
    std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

    cudaMemcpy(R2, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
    cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  }}

  float elapsedTimeT3 = 0.0f;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3); CHECK_ERR;
  std::cout << "cuTensor sub-kernel 2 took: " << elapsedTimeT3 << " ms" << std::endl; 
  std::cout << "cuTensor Tensor Contraction took: " << elapsedTimeT2 + elapsedTimeT3 << " ms" << std::endl; 


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
  std::cout << "Gemmforge GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational intensity: " << fp_per_el / ls_per_el << std::endl;
 
  double peakFLOPGiven = {peakFLOPGiven};
  double peakBandwidthGiven = {peakBandwidthGiven};

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge" << std::endl;
    std::cout << 100.0*(fp_per_el * 1e-6 / (elapsedTimeT2+elapsedTimeT3)) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;
  }}

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] w;
  delete[] A_dev_begins;
  delete[] B_dev_begins;
  delete[] C_dev_begins;
  delete[] w_dev_begins;
  delete[] R1;
  delete[] R2;

  cudaFree(A_dev);
  cudaFree(C_dev);
  cudaFree(w_dev);
  cudaFree(X_dev);
  cudaFree(B_dev);
  currentAllocSize -= sizeof(float) * {sizeA} * num_els +
                      sizeof(float) * {sizeC} * num_els +
                      sizeof(float) * {sizew} * num_els +
                      sizeof(float) * {sizeB} * num_els +
                      sizeof(float) * {shapeB[0]} * {shapeB[2]} * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

  return 0;
}}

"""

  code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_kernel_1_variant_{dimId}.cu", "w")
  code_file.write(benchmark_str)
  code_file.flush()
  code_file.close()
