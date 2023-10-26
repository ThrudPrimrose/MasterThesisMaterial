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
         (31, 73),
         (31, 11)],
        [(48, 72), 
         (62, 16, 48),
         (16,),
         (62, 72),
         (62, 48)],
        [(32, 32), 
         (32, 32, 32),
         (32,),
         (32, 32),
         (32, 32)],
        [(16, 16), 
         (16, 16, 16),
         (16,),
         (16, 16),
         (16, 16)],
        [(3, 11), 
         (5, 7, 3),
         (7,),
         (5, 11),
         (5, 3)],
        [(53, 33), 
         (21, 23, 53),
         (23,),
         (21, 33),
         (21, 53)]]


for dimId in range(6):
  shapeA = dims[dimId][0]
  shapeB = dims[dimId][1]
  shapew = dims[dimId][2]
  shapeC = dims[dimId][3]
  shapeX = dims[dimId][4]
  A = Tensor('A', shapeA)
  B = Tensor('B', shapeB)
  w = Tensor('w', shapew)
  C = Tensor('C', shapeC)
  X = Tensor('X', shapeX)
  sizeA = reduce(operator.mul, shapeA, 1)
  sizeB = reduce(operator.mul, shapeB, 1)
  sizew = reduce(operator.mul, shapew, 1)
  sizeC = reduce(operator.mul, shapeC, 1)
  sizeX = reduce(operator.mul, shapeX, 1)
  maxShapeLen = max(len(shapeA), len(shapeB), len(shapeC), len(shapew), len(shapeX))

  a_tuple = ",".join(str(item) for item in shapeA)
  b_tuple = ",".join(str(item) for item in shapeB)
  c_tuple = ",".join(str(item) for item in shapeC)
  w_tuple = ",".join(str(item) for item in shapew)
  shape_str = ""
  shape_str += f"A({a_tuple}), B({b_tuple}), C({c_tuple}), w({w_tuple})"

  kernel1 = X['il'] <=  B['ikl'] * w['k']
  kernel2 = C['ij'] <=  C['ij'] + A['lj'] * X['il']
  kernel3 = C['ij'] <=  C['ij'] + A['lj'] * B['ikl'] * w['k']

  def get_available_mem_on_gpu():
      meminfo = cuda.current_context().get_memory_info()
      return meminfo[0]


  def get_suggested_num_elements():
      # 1 pointer extra needed per element
      per_el_size = (sizeA + sizeB + sizew + sizeC + sizeX) * 4 + 16

      available_mem = get_available_mem_on_gpu()
      can_fit_els = available_mem // per_el_size
      lower = int(0.80 * can_fit_els)
      return lower

  fp_per_el_0 = 0
  fp_per_el_0 += 11*31*27*2 # Comp loop first kernel
  fp_per_el_0 += 31*11*73*2 # Comp loop second kernel
  fp_per_el_0 += 31*73 # C += second kernel

  fp_per_el_1 = 0
  fp_per_el_1 += 62*48*16*2 # Comp loop first kernel
  fp_per_el_1 += 62*48*72*2 # Comp loop second kernel
  fp_per_el_1 += 62*72 # C += second kernel

  fp_per_el_2 = 0
  fp_per_el_2 += 32*32*32*2 # Comp loop first kernel
  fp_per_el_2 += 32*32*32*2 # Comp loop second kernel
  fp_per_el_2 += 32*32 # C += second kernel

  fp_per_el_3 = 0
  fp_per_el_3 += 16*16*16*2 # Comp loop first kernel
  fp_per_el_3 += 16*16*16*2 # Comp loop second kernel
  fp_per_el_3 += 16*16 # C += second kernel

  fp_per_el_4 = 0
  fp_per_el_4 += 3*5*7*2 # Comp loop first kernel
  fp_per_el_4 += 5*3*11*2 # Comp loop second kernel
  fp_per_el_4 += 5*11 # C += second kernel

  fp_per_el_5 = 0
  fp_per_el_5 += 53*21*23*2 # Comp loop first kernel
  fp_per_el_5 += 21*53*33*2 # Comp loop second kernel
  fp_per_el_5 += 23*33 # C += second kernel

  fp_per_els = [fp_per_el_0, fp_per_el_1, fp_per_el_2, fp_per_el_3, fp_per_el_4, fp_per_el_5]
  fp_per_el = fp_per_els[dimId]

  ls_per_el = 0
  ls_per_el += sizeA # load w first kernel
  ls_per_el += sizeB # load B first kernel
  ls_per_el += sizeC*2 # load A second kernel
  ls_per_el += sizew # load and write C second kernel
  ls_per_el *= 4

  ls_per_els = list()
  for j in dims:
    _shapeA = j[0]
    _shapeB = j[1]
    _shapew = j[2]
    _shapeC = j[3]
    _sizeA = reduce(operator.mul, _shapeA, 1)
    _sizeB = reduce(operator.mul, _shapeB, 1)
    _sizew = reduce(operator.mul, _shapew, 1)
    _sizeC = reduce(operator.mul, _shapeC, 1)
    ls_per_els.append(_sizeA + _sizeB + _sizew + _sizeC*2)


  peakBandwidthGiven = tensorPeakBandwidth
  peakFLOPGiven = tensorPeakFLOP

  num_els = get_suggested_num_elements()

  gpu_kernels = list()
  function_names = list()
  function_argss = list()

  for i, kernel in enumerate([kernel1, kernel2, kernel3]):
    arch = useArchitectureIdentifiedBy(
        host_arch="shsw", device_arch="ssm_86", device_backend="cuda")
    generator = Generator(arch)

    generator.add(name=f'kernel{i+1}', ast=kernel, target="gpu")

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

    gpu_kernels.append(gpu_kernel)#
    function_names.append(function_name)
    function_argss.append(function_args)


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

{gpu_kernels[0]}
{gpu_kernels[1]}
{gpu_kernels[2]}

int main(){{

  constexpr size_t num_els = {num_els};
  float* A = new float[{sizeA} * num_els];
  float* B = new float[{sizeB} * num_els];
  float* C = new float[{sizeC} * num_els];
  float* w = new float[{sizew} * num_els];
  float* R1 = new float[{sizeC} * num_els]{{0.f}};
  float* R2 = new float[{sizeC} * num_els]{{0.f}};
  float* X = new float[{sizeX} * num_els]{{0.f}};

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
  float* X_dev = nullptr;

  float** A_dev_begins = new float*[num_els];
  float** B_dev_begins = new float*[num_els];
  float** C_dev_begins = new float*[num_els];
  float** w_dev_begins = new float*[num_els];
  float** X_dev_begins = new float*[num_els];

  float** A_dev_begins_dev = nullptr;
  float** B_dev_begins_dev = nullptr;
  float** C_dev_begins_dev = nullptr;
  float** w_dev_begins_dev = nullptr;
  float** X_dev_begins_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * {sizeA} * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * {sizeB} * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * {sizeC} * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev, sizeof(float) * {sizew} * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * {sizeX} * num_els); CHECK_ERR;
  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float) * {sizeX} * num_els); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {sizeA} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * {sizeB} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev, (void *)w, sizeof(float) * {sizew} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * {sizeX} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{
    A_dev_begins[i] = A_dev + i * {sizeA};
    B_dev_begins[i] = B_dev + i * {sizeB};
    C_dev_begins[i] = C_dev + i * {sizeC};
    w_dev_begins[i] = w_dev + i * {sizew};
    X_dev_begins[i] = X_dev + i * {sizeX};
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev_begins_dev, (void *)w_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  {function_names[2]}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Will compute the kernel: C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k'], with Gemmforge" << std::endl;
  std::cout << "Shapes and dims: " << "{shape_str}" << std::endl;
  float elapsedTime0 = 0.0; 
  cudaEvent_t startT0, stopT0;
  cudaEventCreate(&startT0); CHECK_ERR;
  cudaEventCreate(&stopT0); CHECK_ERR;
  cudaEventRecord(startT0); CHECK_ERR;
  {function_names[2]}(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT0); CHECK_ERR;
  cudaEventSynchronize(stopT0); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime0, startT0, stopT0); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Fused Tensor Contraction took: " << elapsedTime0 << " ms" << std::endl; 
  cudaMemcpy(R1, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  float elapsedTime1 = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  {function_names[0]}(B_dev_begins_dev, 0, X_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime1, startT1, stopT1); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Unfused Tensor Contraction Part 1 took: " << elapsedTime1 << " ms" << std::endl; 
  
  float elapsedTime2 = 0.0; 
  cudaEvent_t startT2, stopT2;
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventRecord(startT2); CHECK_ERR;
  {function_names[1]}(A_dev_begins_dev, 0, C_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT2); CHECK_ERR;
  cudaEventSynchronize(stopT2); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime2, startT2, stopT2); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Unfused Tensor Contraction Part 2 took: " << elapsedTime2 << " ms" << std::endl; 
  cudaMemcpy(R2, C_dev, sizeof(float) * {sizeC} * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * {sizeC} * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  double fp_per_el = {fp_per_el};
  double ls_per_el = {ls_per_el};
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  std::cout << "Gemmforge GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime0 << std::endl;
  std::cout << "Unfused GFLOPs/s: " << fp_per_el * 1e-6 / (elapsedTime1 + elapsedTime2) << std::endl;
  std::cout << "Operational intensity: " << fp_per_el / ls_per_el << std::endl;
 
  double peakFLOPGiven = {peakFLOPGiven};
  double peakBandwidthGiven = {peakBandwidthGiven};

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){{
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime0) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge Fused" << std::endl;
    std::cout << 100.0*(fp_per_el * 1e-6 / (elapsedTime1+elapsedTime2)) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Unfused" << std::endl;
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

  return 0;
}}

"""

  code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_tensor_kernel_1_variant_{dimId}.cu", "w")
  code_file.write(benchmark_str)
  code_file.flush()
  code_file.close()
