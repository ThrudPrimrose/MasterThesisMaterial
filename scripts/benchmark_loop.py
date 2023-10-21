import os
from functools import reduce
import operator
import subprocess
from yateto import *
from params import *
from numba import cuda

l1s = [16, 24, 1, 48, 12]
l2s = [16, 24, 1, 48, 12]
l3s = [144, 1 , 72]

for l1 in l1s:
  for l2 in l2s:
    for l3 in l3s:

      benchmark_str = f"""
#include <random>
#include <iostream>
#include <cstring>
#include <vector>
#include <unordered_map>

#include <cutensor.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(x)                                                  \
{{                                                                         \
  const auto err = x;                                                    \
  if( err != CUTENSOR_STATUS_SUCCESS )                                   \
  {{                                                                      \
    std::cout << "Error: " << cutensorGetErrorString(err) << std::endl;  \
    return err;                                                          \
  }}                                                                      \
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

__global__ void 
__launch_bounds__(96)
 kernel_sloopOverGEMM_NT_NT_NT_NT_NT_NT__d96_1_d96_48_d48_144_d96_16_d96_144_d16_1__alpha_1_0alpha_1_0_beta_0_0beta_1_0_s_s_p_p_p_p__3310992(const float * const * A, int A_extraOffset, const float * const * B, int B_extraOffset, float ** C, int C_extraOffset, const float * const * w, int w_extraOffset, unsigned numElements, unsigned* flags) {{ 
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {{ 
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {{ 
      __shared__  __align__(8) float _tmp0_buffer_alloc[4608];
      float * _tmp0_buffer = &_tmp0_buffer_alloc[4608 * threadIdx.y];
      float * tmp0 = &_tmp0_buffer[0];
      float * tmp1 = &_tmp0_buffer[0];
      #pragma unroll {l1}
      for (int _l = 0; _l < 48; ++_l) {{ 
        //Original Loop: const float* _A = B + 1536*_l
        //Original Loop: const float* _B = w 
        //Original Loop: float* _C = _tmp0 + 96*_l
        {{ 
        const float * const __restrict__ glb_B = &B[batchID][0 + B_extraOffset + 1536*_l];
          const float * const __restrict__ glb_w = &w[batchID][0 + w_extraOffset];
          float * const __restrict__ _tmp0 = &tmp0[0 + 96*_l];
          float reg0 = 0.0;
          __shared__  __align__(8) float totalShrMem[16];
          float * localShrMem0 = &totalShrMem[16 * threadIdx.y];
          
          float* shrRegion0 = &localShrMem0[0];
          // using ExtendedPatchLoader
          {{ 
            if (threadIdx.x < 16) {{ 
              shrRegion0[threadIdx.x + 0] = glb_w[threadIdx.x + 0];
            }}
          }}
          __syncthreads();
          if (threadIdx.x < 96) {{ 
            float value;
          
            #pragma unroll
            for (int k = 0; k < 16; ++k) {{ 
              value = glb_B[threadIdx.x + k * 96];
          
              #pragma unroll
              for (int n = 0; n < 1; ++n) {{ 
                reg0 += value * shrRegion0[k + 16 * n];
              }}
            }}
          }}
          if (threadIdx.x < 96) {{ 
            _tmp0[threadIdx.x] = reg0;
          }}
          
        }}
      }}
      {{ 
    //('gemm', {{ 'descr': Description(  result=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(96, 144), bbox=BoundingBox(Range(0, 96), Range(0, 144)), stride=(1, 96), align=<yateto.arch.Architecture object at 0x7f19b3b69090>),	  eqspp=dense(shape=(96, 144), size=13824, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=tmp1,	  memoryLayout=DenseMemoryLayout(shape=(96, 48), bbox=BoundingBox(Range(0, 96), Range(0, 48)), stride=(1, 96), align=<yateto.arch.Architecture object at 0x7f19b3b69090>),	  eqspp=dense(shape=(96, 48), size=4608, ndim=2),	  is_compute_constant=False,	  is_temporary=True),	  rightTerm=TensorDescription(  name=A,	  memoryLayout=DenseMemoryLayout(shape=(48, 144), bbox=BoundingBox(Range(0, 48), Range(0, 144)), stride=(1, 48), align=<yateto.arch.Architecture object at 0x7f19b3b69090>),	  eqspp=dense(shape=(48, 144), size=6912, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=True,	  alignedC=True,	  mnk=(Range(0, 96), Range(0, 144), Range(0, 48))), 'matrix_a': DenseMatrix{{ name = tmp1, num. rows = 96, num. columns = 48, leading dimension = 96, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 96, 48]}}, 'matrix_b': DenseMatrix{{ name = A, num. rows = 48, num. columns = 144, leading dimension = 48, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 48, 144]}}, 'matrix_c': DenseMatrix{{ name = C, num. rows = 96, num. columns = 144, leading dimension = 96, direction = DataFlowDirection.SINK, bbox = [0, 0, 96, 144]}}, 'args': ['_tmp0, 0', 'A, extraOffset_A', 'C, extraOffset_C', 'numElements', 'flags', 'streamPtr']}})
        const float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        float * const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        const float * const __restrict__ _tmp1 = &tmp1[0];
        float reg0[144] = {{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
        __shared__  __align__(8) float totalShrMem[6912];
        float * localShrMem0 = &totalShrMem[6912 * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {{ 
          #pragma unroll
          for (int i = 0; i < 72; ++i) {{ 
            shrRegion0[threadIdx.x + i * 96] = glb_A[threadIdx.x + i * 96];
          }}
        }}
        __syncthreads();
        if (threadIdx.x < 96) {{ 
          float value;
        
          #pragma unroll {l2}
          for (int k = 0; k < 48; ++k) {{ 
            value = _tmp1[threadIdx.x + k * 96];
        
            #pragma unroll {l3}
            for (int n = 0; n < 144; ++n) {{ 
              reg0[n] += value * shrRegion0[k + 48 * n];
            }}
          }}
        }}
        if (threadIdx.x < 96) {{ 
          #pragma unroll
          for (int n = 0; n < 144; ++n) {{ 
            glb_C[threadIdx.x + 96 * n] = reg0[n] + glb_C[threadIdx.x + 96 * n];
          }}
        }}
        
      }}
    }}
  }}
}}
void sloopOverGEMM_NT_NT_NT_NT_NT_NT__d96_1_d96_48_d48_144_d96_16_d96_144_d16_1__alpha_1_0alpha_1_0_beta_0_0beta_1_0_s_s_p_p_p_p__3310992(const float * const * A, int A_extraOffset, const float * const * B, int B_extraOffset, float ** C, int C_extraOffset, const float * const * w, int w_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {{ 
  dim3 block(96, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT_NT_NT_NT__d96_1_d96_48_d48_144_d96_16_d96_144_d16_1__alpha_1_0alpha_1_0_beta_0_0beta_1_0_s_s_p_p_p_p__3310992<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, w, w_extraOffset, numElements, flags);
  CHECK_ERR;
}}




int main(){{ 
  size_t currentAllocSize = 0;

  constexpr size_t num_els = 7420;
  float* A = new float[6912 * num_els];
  float* B = new float[73728 * num_els];
  float* C = new float[13824 * num_els];
  float* w = new float[16 * num_els];
  float* R1 = new float[13824 * num_els]{{ 0.f}};
  float* R2 = new float[13824 * num_els]{{ 0.f}};

  float coreA[6912];
  float coreB[73728];
  float coreC[13824];
  float corew[16];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < 6912; i++){{ 
    coreA[i] = distribution(gen);
  }}
  for (size_t i = 0; i < 73728; i++){{ 
    coreB[i] = distribution(gen);
  }}
  for (size_t i = 0; i < 13824; i++){{ 
    coreC[i] = distribution(gen);
  }}
  for (size_t i = 0; i < 16; i++){{ 
    corew[i] = distribution(gen);
  }}

  for (size_t i = 0; i < num_els; i++){{ 
      std::memcpy(&A[i * 6912], &coreA[0], 6912 * sizeof(float));
      std::memcpy(&B[i * 73728], &coreB[0], 73728 * sizeof(float));
      std::memcpy(&C[i * 13824], &coreC[0], 13824 * sizeof(float));
      std::memcpy(&w[i * 16], &corew[0], 16 * sizeof(float));
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

  cudaMalloc((void **)&A_dev, sizeof(float) * 6912 * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * 73728 * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * 13824 * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev, sizeof(float) * 16 * num_els); CHECK_ERR;
  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&w_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  currentAllocSize += sizeof(float) * 6912 * num_els +
                      sizeof(float) * 73728 * num_els +
                      sizeof(float) * 13824 * num_els +
                      sizeof(float) * 16 * num_els +
                      4 * sizeof(float*) * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 6912 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 73728 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 13824 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev, (void *)w, sizeof(float) * 16 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){{ 
    A_dev_begins[i] = A_dev + i * 6912;
    B_dev_begins[i] = B_dev + i * 73728;
    C_dev_begins[i] = C_dev + i * 13824;
    w_dev_begins[i] = w_dev + i * 16;
  }}

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)w_dev_begins_dev, (void *)w_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  sloopOverGEMM_NT_NT_NT_NT_NT_NT__d96_1_d96_48_d48_144_d96_16_d96_144_d16_1__alpha_1_0alpha_1_0_beta_0_0beta_1_0_s_s_p_p_p_p__3310992(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 13824 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Will compute the kernel: C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k'], with Gemmforge" << std::endl;
  std::cout << "Shapes and dims: " << "A(48,144), B(96,16,48), C(96,144), w(16)" << std::endl;
  float elapsedTime = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  sloopOverGEMM_NT_NT_NT_NT_NT_NT__d96_1_d96_48_d48_144_d96_16_d96_144_d16_1__alpha_1_0alpha_1_0_beta_0_0beta_1_0_s_s_p_p_p_p__3310992(A_dev_begins_dev, 0, B_dev_begins_dev, 0, C_dev_begins_dev, 0, w_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startT1, stopT1); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, C_dev, sizeof(float) * 13824 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 13824 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  std::cout << "Will compute the kernel: C['ij'] <= C['ij'] + A['lj'] * B['ikl'] * w['k'], with cuTensor" << std::endl;
  std::cout << "Need to split into 2 kernels, 1: X['il'] <= B['ikl'] * w['k'], with cuTensor" << std::endl;
  std::cout <<"Need to split into 2 kernels, 2: C['ij'] <=  A['lj'] * X['il'], with cuTensor" << std::endl;
  std::cout << "Batched version managed through: C['ijb'] <= C['ijb'] + A['ljb'] * B['iklb'] * w['kb'], with cuTensor" << std::endl;

  float* X_dev = nullptr;
  cudaMalloc((void **)&X_dev, sizeof(float) * 96 * 48 * num_els); CHECK_ERR;
  currentAllocSize += sizeof(float) * 96 * 48 * num_els;
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
    std::vector<int> modeX{{ 'i', 'l', 'b'}};
    std::vector<int> modeB{{ 'i', 'k', 'l', 'b'}};
    std::vector<int> modew{{ 'k', 'b'}};
    int nmodeX = modeX.size();
    int nmodeB = modeB.size();
    int nmodew = modew.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = 96;
    extent['k'] = 16;
    extent['l'] = 48;
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
    cudaMemset(X_dev, 0.0f, 1536 * num_els); CHECK_ERR;

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
    std::vector<int> modeA{{ 'l', 'j', 'b'}};
    std::vector<int> modeC{{ 'i', 'j', 'b'}};
    std::vector<int> modeX{{ 'i', 'l', 'b'}};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();
    int nmodeX = modeX.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['i'] = 96;
    extent['j'] = 144;
    extent['l'] = 48;
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
    cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 13824 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

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

    cudaMemcpy(R2, C_dev, sizeof(float) * 13824 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
    cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 13824 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  }}

  float elapsedTimeT3 = 0.0f;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3); CHECK_ERR;
  std::cout << "cuTensor sub-kernel 2 took: " << elapsedTimeT3 << " ms" << std::endl; 
  std::cout << "cuTensor Tensor Contraction took: " << elapsedTimeT2 + elapsedTimeT3 << " ms" << std::endl; 


  bool results_wrong = false;
  for (size_t i = 0; i < 13824 * num_els; i++){{ 
    if (std::abs(R1[i] - R2[i]) > 1.0f) {{ 
      std::cout << "Results do not match, problem first at offset " << i << " :_(" << std::endl;
      results_wrong = true;
      break;
    }}
  }}
  if (!results_wrong){{ 
    std::cout << "Gemmforge and cuTensor contraction results match! :)" << std::endl;
  }}

  double fp_per_el = 1488384;
  double ls_per_el = 433216;
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  std::cout << "Gemmforge GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational intensity: " << fp_per_el / ls_per_el << std::endl;
 
  double peakFLOPGiven = 4329.47;
  double peakBandwidthGiven = 176.032;

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
  currentAllocSize -= sizeof(float) * 6912 * num_els +
                      sizeof(float) * 13824 * num_els +
                      sizeof(float) * 16 * num_els +
                      sizeof(float) * 73728 * num_els +
                      sizeof(float) * 96 * 48 * num_els;
  std::cout << "Current Device Alloc Size: " << static_cast<float>(currentAllocSize) / (1024.0 * 1024.0 * 1024.0) << std::endl;

  return 0;
}}


"""

      code_file = open(f"{scripts_dir}/cuda_code/benchmark_cuda_loop_{l1}-{l2}-{l3}.cu", "w")
      code_file.write(benchmark_str)
      code_file.flush()
      code_file.close()

