
#include <random>
#include <iostream>
#include <cstring>
#include <vector>
#include <unordered_map>

#include <cutensor.h>
#include <cuda_runtime.h>

#define HANDLE_ERROR(x)                                                  \
{                                                                        \
  const auto err = x;                                                    \
  if( err != CUTENSOR_STATUS_SUCCESS )                                   \
  {                                                                      \
    std::cout << "Error: " << cutensorGetErrorString(err) << std::endl;  \
    std::cout << __FILE__ << " " << __LINE__ << std::endl;                      \
  }                                                                      \
}

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

std::string PrevFile = "";
int PrevLine = 0;

void checkErr(const std::string &File, int Line) {
#ifndef NDEBUG
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess) {
        std::cout << std::endl << File
                << ", line " << Line
                << ": " << cudaGetErrorString(Error)
                << " (" << Error << ")"
                << std::endl;

        if (PrevLine > 0)
        std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
    }
    PrevFile = File;
    PrevLine = Line;
#endif
}

__global__ void 
__launch_bounds__(32)
 kernel_sloopOverGEMM_NT_NT_NT__d8_15_d8_13_d15_13__alpha_1_0_beta_0_0_p_p_p__63aca98(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(8, 13), bbox=BoundingBox(Range(0, 8), Range(0, 13)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(8, 13), size=104, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(8, 15), bbox=BoundingBox(Range(0, 8), Range(0, 15)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(8, 15), size=120, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(15, 13), bbox=BoundingBox(Range(0, 15), Range(0, 13)), stride=(1, 15), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(15, 13), size=195, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 8), Range(0, 13), Range(0, 15))), 'matrix_a': DenseMatrix{name = C, num. rows = 8, num. columns = 15, leading dimension = 8, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 8, 15]}, 'matrix_b': DenseMatrix{name = D, num. rows = 15, num. columns = 13, leading dimension = 15, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 15, 13]}, 'matrix_c': DenseMatrix{name = X, num. rows = 8, num. columns = 13, leading dimension = 8, direction = DataFlowDirection.SINK, bbox = [0, 0, 8, 13]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(8, 13), bbox=BoundingBox(Range(0, 8), Range(0, 13)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(8, 13), size=104, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(8, 15), bbox=BoundingBox(Range(0, 8), Range(0, 15)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(8, 15), size=120, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(15, 13), bbox=BoundingBox(Range(0, 15), Range(0, 13)), stride=(1, 15), align=<yateto.arch.Architecture object at 0x7fbd191e6410>),	  eqspp=dense(shape=(15, 13), size=195, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 8), Range(0, 13), Range(0, 15))), 'matrix_a': DenseMatrix{name = C, num. rows = 8, num. columns = 15, leading dimension = 8, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 8, 15]}, 'matrix_b': DenseMatrix{name = D, num. rows = 15, num. columns = 13, leading dimension = 15, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 15, 13]}, 'matrix_c': DenseMatrix{name = X, num. rows = 8, num. columns = 13, leading dimension = 8, direction = DataFlowDirection.SINK, bbox = [0, 0, 8, 13]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
        const float * const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        const float * const __restrict__ glb_D = &D[batchID][0 + D_extraOffset];
        float reg0[13] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[195];
        float * localShrMem0 = &totalShrMem[195 * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 6; ++i) {
            shrRegion0[threadIdx.x + i * 32] = glb_D[threadIdx.x + i * 32];
          }
          if (threadIdx.x < 3) {
            shrRegion0[threadIdx.x + 192] = glb_D[threadIdx.x + 192];
          }
        }
        __syncwarp();
        if (threadIdx.x < 8) {
          float value;
        
          #pragma unroll
          for (int k = 0; k < 15; ++k) {
            value = glb_C[threadIdx.x + k * 8];
        
            #pragma unroll
            for (int n = 0; n < 13; ++n) {
              reg0[n] += value * shrRegion0[k + 15 * n];
            }
          }
        }
        if (threadIdx.x < 8) {
          #pragma unroll
          for (int n = 0; n < 13; ++n) {
            glb_X[threadIdx.x + 8 * n] = reg0[n];
          }
        }
        
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d8_15_d8_13_d15_13__alpha_1_0_beta_0_0_p_p_p__63aca98(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d8_15_d8_13_d15_13__alpha_1_0_beta_0_0_p_p_p__63aca98<<<grid,block,0,stream>>>(C, C_extraOffset, D, D_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}


__global__ void 
__launch_bounds__(128)
 kernel_sloopOverGEMM_NT_NT_NT__d104_46_d14_46_d104_14__alpha_1_0_beta_1_0_p_p_p__5ce9ba2(float ** A, int A_extraOffset, const float * const * E, int E_extraOffset, const float * const * F, int F_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=A,	  memoryLayout=DenseMemoryLayout(shape=(104, 46), bbox=BoundingBox(Range(0, 104), Range(0, 46)), stride=(1, 104), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(104, 46), size=4784, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=E,	  memoryLayout=DenseMemoryLayout(shape=(104, 14), bbox=BoundingBox(Range(0, 104), Range(0, 14)), stride=(1, 104), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(104, 14), size=1456, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=F,	  memoryLayout=DenseMemoryLayout(shape=(14, 46), bbox=BoundingBox(Range(0, 14), Range(0, 46)), stride=(1, 14), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(14, 46), size=644, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 104), Range(0, 46), Range(0, 14))), 'matrix_a': DenseMatrix{name = E, num. rows = 104, num. columns = 14, leading dimension = 104, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 104, 14]}, 'matrix_b': DenseMatrix{name = F, num. rows = 14, num. columns = 46, leading dimension = 14, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 14, 46]}, 'matrix_c': DenseMatrix{name = A, num. rows = 104, num. columns = 46, leading dimension = 104, direction = DataFlowDirection.SINK, bbox = [0, 0, 104, 46]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F', 'A, extraOffset_A', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=A,	  memoryLayout=DenseMemoryLayout(shape=(104, 46), bbox=BoundingBox(Range(0, 104), Range(0, 46)), stride=(1, 104), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(104, 46), size=4784, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=E,	  memoryLayout=DenseMemoryLayout(shape=(104, 14), bbox=BoundingBox(Range(0, 104), Range(0, 14)), stride=(1, 104), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(104, 14), size=1456, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=F,	  memoryLayout=DenseMemoryLayout(shape=(14, 46), bbox=BoundingBox(Range(0, 14), Range(0, 46)), stride=(1, 14), align=<yateto.arch.Architecture object at 0x7fbd191d77d0>),	  eqspp=dense(shape=(14, 46), size=644, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 104), Range(0, 46), Range(0, 14))), 'matrix_a': DenseMatrix{name = E, num. rows = 104, num. columns = 14, leading dimension = 104, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 104, 14]}, 'matrix_b': DenseMatrix{name = F, num. rows = 14, num. columns = 46, leading dimension = 14, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 14, 46]}, 'matrix_c': DenseMatrix{name = A, num. rows = 104, num. columns = 46, leading dimension = 104, direction = DataFlowDirection.SINK, bbox = [0, 0, 104, 46]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F', 'A, extraOffset_A', 'numElements', 'flags', 'streamPtr']})
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float * const __restrict__ glb_F = &F[batchID][0 + F_extraOffset];
        const float * const __restrict__ glb_E = &E[batchID][0 + E_extraOffset];
        float reg0[46] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[644];
        float * localShrMem0 = &totalShrMem[644 * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 5; ++i) {
            shrRegion0[threadIdx.x + i * 128] = glb_F[threadIdx.x + i * 128];
          }
          if (threadIdx.x < 4) {
            shrRegion0[threadIdx.x + 640] = glb_F[threadIdx.x + 640];
          }
        }
        __syncthreads();
        if (threadIdx.x < 104) {
          float value;
        
          #pragma unroll
          for (int k = 0; k < 14; ++k) {
            value = glb_E[threadIdx.x + k * 104];
        
            #pragma unroll
            for (int n = 0; n < 46; ++n) {
              reg0[n] += value * shrRegion0[k + 14 * n];
            }
          }
        }
        if (threadIdx.x < 104) {
          #pragma unroll
          for (int n = 0; n < 46; ++n) {
            glb_A[threadIdx.x + 104 * n] = reg0[n] + glb_A[threadIdx.x + 104 * n];
          }
        }
        
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d104_46_d14_46_d104_14__alpha_1_0_beta_1_0_p_p_p__5ce9ba2(float ** A, int A_extraOffset, const float * const * E, int E_extraOffset, const float * const * F, int F_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(128, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d104_46_d14_46_d104_14__alpha_1_0_beta_1_0_p_p_p__5ce9ba2<<<grid,block,0,stream>>>(A, A_extraOffset, E, E_extraOffset, F, F_extraOffset, numElements, flags);
  CHECK_ERR;
}


__global__ void 
__launch_bounds__(384)
 kernel_sproduct_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      {
        const float * const __restrict__ glb_B = &B[batchID][0 + B_extraOffset];
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float reg0[13] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[150];
        float * localShrMem0 = &totalShrMem[150 * threadIdx.y];

        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedTensorLoader
        {
          if (threadIdx.x < 46) {
            shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
          }
        }

        float* shrRegion1 = &localShrMem0[46];
        // using ExtendedTensorLoader
        {
          if (threadIdx.x < 104) {
            shrRegion1[threadIdx.x + 0] = glb_X[threadIdx.x + 0];
          }
        }
        __syncthreads();
        /*
        This is the product kernel created from the following YaTeTo description:
        Description(
        	alpha: 1.0
        	add: True
        	result: IndexedTensorDescription(name=A, indices=kpm, memoryLayout=DenseMemoryLayout(shape=(8, 13, 46), bbox=BoundingBox(Range(0, 8), Range(0, 13), Range(0, 46)), stride=(1, 8, 104), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13, 46), size=4784, ndim=3), is_compute_constant=False, is_temporary=False)
        	leftTerm: IndexedTensorDescription(name=B, indices=m, memoryLayout=DenseMemoryLayout(shape=(46,), bbox=BoundingBox(Range(0, 46)), stride=(1,), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(46,), size=46, ndim=1), is_compute_constant=False, is_temporary=False)
        	rightTerm: IndexedTensorDescription(name=X, indices=kp, memoryLayout=DenseMemoryLayout(shape=(8, 13), bbox=BoundingBox(Range(0, 8), Range(0, 13)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13), size=104, ndim=2), is_compute_constant=False, is_temporary=False)
        	isACsc: False
        	isBCsc: False
        	loopRanges: {'m': Range(0, 46), 'p': Range(0, 13), 'k': Range(0, 8)}
        )
        */
        if (threadIdx.x < 368) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 8;
          rows_left -= row_offset_1 * 8;
          const int dim_offset_m = row_offset_1;
          const int row_offset_0 = rows_left;
          const int dim_offset_k = row_offset_0;
          #pragma unroll
          for (int p = 0; p < 13; ++p) {
            reg0[p] = shrRegion0[dim_offset_m * 1] * shrRegion1[dim_offset_k * 1 + p * 8];
          }
        }
        if (threadIdx.x < 368) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 8;
          rows_left -= row_offset_1 * 8;
          const int row_offset_0 = rows_left;
          #pragma unroll
          for (int i = 0; i < 13; ++i) {
            glb_A[row_offset_0 * 1 + row_offset_1 * 104 + i * 8] = reg0[i] + 1.0 * glb_A[row_offset_0 * 1 + row_offset_1 * 104 + i * 8];
          }
        }
      }
    }
  }
}
void sproduct_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(384, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sproduct_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}

__global__ void 
__launch_bounds__(384)
 kernel_sproduct2_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      {
        const float * const __restrict__ glb_B = &B[batchID][0 + B_extraOffset];
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float reg0[13] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[150 + 4784];
        float * localShrMem0 = &totalShrMem[(150 + 4784) * threadIdx.y];

        float* shrRegion0 = &localShrMem0[0];
        float* shrRegion1 = &localShrMem0[46];
        float* shrRegion2 = &localShrMem0[150];
        // using ExtendedTensorLoader
        float* loadRegion = (threadIdx.x < 46)? shrRegion0 : shrRegion1;
        const float* globalregion = (threadIdx.x < 46)? glb_B : glb_X;
        int loadSubstract = (threadIdx.x < 46)? 0 : -46;
        {
          if (threadIdx.x < 150) {
            loadRegion[threadIdx.x + loadSubstract] = globalregion[threadIdx.x + loadSubstract];
          }

          for (int i = 0; i < 12; i++){
            shrRegion2[threadIdx.x + i*384] = glb_A[threadIdx.x + i*384];
          }

          if (threadIdx.x < 176){
            shrRegion2[threadIdx.x + 12*384] = glb_A[threadIdx.x + 12*384];
          }
        }

        __syncthreads();
        /*
        This is the product kernel created from the following YaTeTo description:
        Description(
        	alpha: 1.0
        	add: True
        	result: IndexedTensorDescription(name=A, indices=kpm, memoryLayout=DenseMemoryLayout(shape=(8, 13, 46), bbox=BoundingBox(Range(0, 8), Range(0, 13), Range(0, 46)), stride=(1, 8, 104), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13, 46), size=4784, ndim=3), is_compute_constant=False, is_temporary=False)
        	leftTerm: IndexedTensorDescription(name=B, indices=m, memoryLayout=DenseMemoryLayout(shape=(46,), bbox=BoundingBox(Range(0, 46)), stride=(1,), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(46,), size=46, ndim=1), is_compute_constant=False, is_temporary=False)
        	rightTerm: IndexedTensorDescription(name=X, indices=kp, memoryLayout=DenseMemoryLayout(shape=(8, 13), bbox=BoundingBox(Range(0, 8), Range(0, 13)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13), size=104, ndim=2), is_compute_constant=False, is_temporary=False)
        	isACsc: False
        	isBCsc: False
        	loopRanges: {'m': Range(0, 46), 'p': Range(0, 13), 'k': Range(0, 8)}
        )
        */
        if (threadIdx.x < 368) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 8;
          rows_left -= row_offset_1 * 8;
          const int row_offset_0 = rows_left;
          const int dim_offset_m = row_offset_1;
          const int dim_offset_k = row_offset_0;
          #pragma unroll
          for (int p = 0; p < 13; ++p) {
            reg0[p] = shrRegion0[dim_offset_m * 1] * shrRegion1[dim_offset_k * 1 + p * 8];
          }

          #pragma unroll
          for (int i = 0; i < 13; ++i) {
            glb_A[row_offset_0 * 1 + row_offset_1 * 104 + i * 8] = reg0[i] + shrRegion2[row_offset_0 * 1 + row_offset_1 * 104 + i * 8];
          }
        }
      }
    }
  }
}
void sproduct2_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(384, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sproduct2_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}

__global__ void 
__launch_bounds__(384)
 kernel_sproduct3_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      {
        const float * const __restrict__ glb_B = &B[batchID][0 + B_extraOffset];
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float reg0[13] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[150];
        float * localShrMem0 = &totalShrMem[150 * threadIdx.y];

        float* shrRegion0 = &localShrMem0[0];
        float* shrRegion1 = &localShrMem0[46];
        // using ExtendedTensorLoader
        float* loadRegion = (threadIdx.x < 46)? shrRegion0 : shrRegion1;
        const float* globalregion = (threadIdx.x < 46)? glb_B : glb_X;
        int loadSubstract = (threadIdx.x < 46)? 0 : -46;
        {
          if (threadIdx.x < 150) {
            loadRegion[threadIdx.x + loadSubstract] = globalregion[threadIdx.x + loadSubstract];
          }
        }
        __syncthreads();
        /*
        This is the product kernel created from the following YaTeTo description:
        Description(
        	alpha: 1.0
        	add: True
        	result: IndexedTensorDescription(name=A, indices=kpm, memoryLayout=DenseMemoryLayout(shape=(8, 13, 46), bbox=BoundingBox(Range(0, 8), Range(0, 13), Range(0, 46)), stride=(1, 8, 104), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13, 46), size=4784, ndim=3), is_compute_constant=False, is_temporary=False)
        	leftTerm: IndexedTensorDescription(name=B, indices=m, memoryLayout=DenseMemoryLayout(shape=(46,), bbox=BoundingBox(Range(0, 46)), stride=(1,), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(46,), size=46, ndim=1), is_compute_constant=False, is_temporary=False)
        	rightTerm: IndexedTensorDescription(name=X, indices=kp, memoryLayout=DenseMemoryLayout(shape=(8, 13), bbox=BoundingBox(Range(0, 8), Range(0, 13)), stride=(1, 8), align=<yateto.arch.Architecture object at 0x7fbd1919e910>), eqspp=dense(shape=(8, 13), size=104, ndim=2), is_compute_constant=False, is_temporary=False)
        	isACsc: False
        	isBCsc: False
        	loopRanges: {'m': Range(0, 46), 'p': Range(0, 13), 'k': Range(0, 8)}
        )
        */
        if (threadIdx.x < 368) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 8;
          rows_left -= row_offset_1 * 8;
          const int dim_offset_m = row_offset_1;
          const int row_offset_0 = rows_left;
          const int dim_offset_k = row_offset_0;
          #pragma unroll
          for (int p = 0; p < 13; ++p) {
            reg0[p] = shrRegion0[dim_offset_m * 1] * shrRegion1[dim_offset_k * 1 + p * 8];
          }
        }
        if (threadIdx.x < 368) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 8;
          rows_left -= row_offset_1 * 8;
          const int row_offset_0 = rows_left;
          #pragma unroll
          for (int i = 0; i < 13; ++i) {
            glb_A[row_offset_0 * 1 + row_offset_1 * 104 + i * 8] = reg0[i] + 1.0 * glb_A[row_offset_0 * 1 + row_offset_1 * 104 + i * 8];
          }
        }
      }
    }
  }
}
void sproduct3_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(384, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sproduct_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}

int main(){
  constexpr size_t num_els = 122554;
  float* A = new float[4784 * num_els]{0.f};
  float* B = new float[46 * num_els]{0.f};
  float* C = new float[120 * num_els]{0.f};
  float* D = new float[195 * num_els]{0.f};
  float* E = new float[1456 * num_els]{0.f};
  float* F = new float[644 * num_els]{0.f};
  float* X = new float[104 * num_els]{0.f};
  float* R1 = new float[4784 * num_els]{0.f};
  float* R2 = new float[4784 * num_els]{0.f};
  //float* Ri1 = new float[104 * num_els]{0.f};
  //float* Ri2 = new float[4784 * num_els]{0.f};
  //float* Ri1c = new float[104 * num_els]{0.f};
  //float* Ri2c = new float[4784 * num_els]{0.f};


  float* coreA = new float[4784];
  float* coreB = new float[46];
  float* coreC = new float[120];
  float* coreD = new float[195];
  float* coreE = new float[1456];
  float* coreF = new float[644];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < 4784; i++){
    coreA[i] = distribution(gen);
  }
  for (size_t i = 0; i < 46; i++){
    coreB[i] = distribution(gen);
  }
  for (size_t i = 0; i < 120; i++){
    coreC[i] = distribution(gen);
  }
  for (size_t i = 0; i < 195; i++){
    coreD[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1456; i++){
    coreE[i] = distribution(gen);
  }
  for (size_t i = 0; i < 644; i++){
    coreF[i] = distribution(gen);
  }

  for (size_t i = 0; i < num_els; i++){
      std::memcpy(&A[i * 4784], &coreA[0], 4784 * sizeof(float));
      std::memcpy(&B[i * 46], &coreB[0], 46 * sizeof(float));
      std::memcpy(&C[i * 120], &coreC[0], 120 * sizeof(float));
      std::memcpy(&D[i * 195], &coreD[0], 195 * sizeof(float));
      std::memcpy(&E[i * 1456], &coreE[0], 1456 * sizeof(float));
      std::memcpy(&F[i * 644], &coreF[0], 644 * sizeof(float));
  }

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

  cudaMalloc((void **)&A_dev, sizeof(float) * 4784 * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * 46 * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * 120 * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * 195 * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * 1456 * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * 644 * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * 104 * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
 
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4784 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 46 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 120 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * 195 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * 1456 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * 644 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 104 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){
    A_dev_begins[i] = A_dev + i * 4784;
    B_dev_begins[i] = B_dev + i * 46;
    C_dev_begins[i] = C_dev + i * 120;
    D_dev_begins[i] = D_dev + i * 195;
    E_dev_begins[i] = E_dev + i * 1456;
    F_dev_begins[i] = F_dev + i * 644;
    X_dev_begins[i] = X_dev + i * 104;
  }

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  sloopOverGEMM_NT_NT_NT__d8_15_d8_13_d15_13__alpha_1_0_beta_0_0_p_p_p__63aca98(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 104 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Dimensions: " << 8 << ", " << 14 << ", " << 46 << ", " << 13 << ", " << 15 << ", " << 14 << std::endl;

  float elapsedTimeT1 = 0.0;
  float elapsedTimeT2 = 0.0;
  float elapsedTimeT3 = 0.0; 
  float elapsedTimeT4 = 0.0;
  cudaEvent_t startT1, stopT1;
  cudaEvent_t startT2, stopT2;
  cudaEvent_t startT3, stopT3;
  cudaEvent_t startT4, stopT4;

  cudaEventCreate(&startT3); CHECK_ERR;
  cudaEventCreate(&stopT3); CHECK_ERR;
  cudaEventRecord(startT3); CHECK_ERR;
  sproduct_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT3); CHECK_ERR;
  cudaEventSynchronize(stopT3); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3); CHECK_ERR;
  //double elapsedTime = elapsedTimeT1 + elapsedTimeT2 + elapsedTimeT3;
  cudaDeviceSynchronize(); CHECK_ERR;
  
  //std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, A_dev, sizeof(float) * 4784 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4784 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventRecord(startT2); CHECK_ERR;
  sproduct3_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT2); CHECK_ERR;
  cudaEventSynchronize(stopT2); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy(R1, A_dev, sizeof(float) * 4784 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4784 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  cudaEventCreate(&startT4); CHECK_ERR;
  cudaEventCreate(&stopT4); CHECK_ERR;
  cudaEventRecord(startT4); CHECK_ERR;
  sproduct2_NT_NT_NT__d46_d8_13_46_d8_13__alpha_1_0_p_p_p__7c7cd48(A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT4); CHECK_ERR;
  cudaEventSynchronize(stopT4); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT4, startT4, stopT4); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy(R2, A_dev, sizeof(float) * 4784 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4784 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;



  double fp_per_el = 156208;
  double ls_per_el = 48116;
  double fp_unfused_per_el = 156208;
  double ls_unfused_per_el = 87220;
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  fp_unfused_per_el *= num_els;
  ls_unfused_per_el *= num_els;
  //std::cout << "Gemmforge Theoretical Fused Kernel GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  //std::cout << "Operational Theoretical Fused intensity: " << fp_per_el / ls_per_el << std::endl;
  //std::cout << "Gemmforge GFLOPs/s: " << fp_unfused_per_el * 1e-6 / elapsedTime << std::endl;
  //std::cout << "Operational intensity: " << fp_unfused_per_el / ls_unfused_per_el << std::endl;
  double peakFLOPGiven = 29767.7;
  double peakBandwidthGiven = 760.08;

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){
    double obtainable_unfused_peak_k3 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(14352) / static_cast<double>(38872)));
    std::cout << 100.0*(14352 * num_els * 1e-6 / elapsedTimeT3) / obtainable_unfused_peak_k3 << " % of roof w. respect to Kernel3 intensity achieved with Gemmforge" << std::endl;
    std::cout << 100.0*(14352 * num_els * 1e-6 / elapsedTimeT2) / obtainable_unfused_peak_k3 << " % of roof w. respect to Kernel3 (Optimization Idea 1) intensity achieved with Gemmforge" << std::endl;
    std::cout << 100.0*(14352 * num_els * 1e-6 / elapsedTimeT4) / obtainable_unfused_peak_k3 << " % of roof w. respect to Kernel3 (Optimization Idea 2) intensity achieved with Gemmforge" << std::endl;
  }

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4784 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 104 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  bool results_wrong = false;
  for (size_t i = 0; i < 4784 * num_els; i++){
    if (std::abs(R1[i] - R2[i]) > 5.0f) {
      std::cout << "Results do not match, problem first at offset " << i << " :_(" << std::endl;
      results_wrong = true;
      break;
    }
  }
  if (!results_wrong){
    std::cout << "Gemmforge and Gemmforge Optimized contraction results match! :)" << std::endl;
  }

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
}

