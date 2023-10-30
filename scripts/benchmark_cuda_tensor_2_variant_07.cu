
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
 kernel_sloopOverGEMM_NT_NT_NT__d10_32_d32_10_d10_10__alpha_1_0_beta_0_0_p_p_p__6a7ea95(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(10, 10), bbox=BoundingBox(Range(0, 10), Range(0, 10)), stride=(1, 10), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(10, 10), size=100, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(10, 32), bbox=BoundingBox(Range(0, 10), Range(0, 32)), stride=(1, 10), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(10, 32), size=320, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(32, 10), bbox=BoundingBox(Range(0, 32), Range(0, 10)), stride=(1, 32), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(32, 10), size=320, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 10), Range(0, 10), Range(0, 32))), 'matrix_a': DenseMatrix{name = C, num. rows = 10, num. columns = 32, leading dimension = 10, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 10, 32]}, 'matrix_b': DenseMatrix{name = D, num. rows = 32, num. columns = 10, leading dimension = 32, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 32, 10]}, 'matrix_c': DenseMatrix{name = X, num. rows = 10, num. columns = 10, leading dimension = 10, direction = DataFlowDirection.SINK, bbox = [0, 0, 10, 10]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(10, 10), bbox=BoundingBox(Range(0, 10), Range(0, 10)), stride=(1, 10), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(10, 10), size=100, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(10, 32), bbox=BoundingBox(Range(0, 10), Range(0, 32)), stride=(1, 10), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(10, 32), size=320, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(32, 10), bbox=BoundingBox(Range(0, 32), Range(0, 10)), stride=(1, 32), align=<yateto.arch.Architecture object at 0x7f3bda32fb50>),	  eqspp=dense(shape=(32, 10), size=320, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 10), Range(0, 10), Range(0, 32))), 'matrix_a': DenseMatrix{name = C, num. rows = 10, num. columns = 32, leading dimension = 10, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 10, 32]}, 'matrix_b': DenseMatrix{name = D, num. rows = 32, num. columns = 10, leading dimension = 32, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 32, 10]}, 'matrix_c': DenseMatrix{name = X, num. rows = 10, num. columns = 10, leading dimension = 10, direction = DataFlowDirection.SINK, bbox = [0, 0, 10, 10]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
        const float * const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        const float * const __restrict__ glb_D = &D[batchID][0 + D_extraOffset];
        float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float reg0[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[640];
        float * localShrMem0 = &totalShrMem[640 * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 10; ++i) {
            shrRegion0[threadIdx.x + i * 32] = glb_C[threadIdx.x + i * 32];
          }
        }
        
        float* shrRegion1 = &localShrMem0[320];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 10; ++i) {
            shrRegion1[threadIdx.x + i * 32] = glb_D[threadIdx.x + i * 32];
          }
        }
        __syncwarp();
        if (threadIdx.x < 10) {
          float value;
        
          #pragma unroll
          for (int k = 0; k < 32; ++k) {
            value = shrRegion0[threadIdx.x + k * 10];
        
            #pragma unroll
            for (int n = 0; n < 10; ++n) {
              reg0[n] += value * shrRegion1[k + 32 * n];
            }
          }
        }
        if (threadIdx.x < 10) {
          #pragma unroll
          for (int n = 0; n < 10; ++n) {
            glb_X[threadIdx.x + 10 * n] = reg0[n];
          }
        }
        
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d10_32_d32_10_d10_10__alpha_1_0_beta_0_0_p_p_p__6a7ea95(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d10_32_d32_10_d10_10__alpha_1_0_beta_0_0_p_p_p__6a7ea95<<<grid,block,0,stream>>>(C, C_extraOffset, D, D_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}


__global__ void 
__launch_bounds__(128)
 kernel_sloopOverGEMM_NT_NT_NT__d17_45_d100_45_d100_17__alpha_1_0_beta_1_0_p_p_p__4972d7a(float ** A, int A_extraOffset, const float * const * E, int E_extraOffset, const float * const * F, int F_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=A,	  memoryLayout=DenseMemoryLayout(shape=(100, 45), bbox=BoundingBox(Range(0, 100), Range(0, 45)), stride=(1, 100), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(100, 45), size=4500, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=E,	  memoryLayout=DenseMemoryLayout(shape=(100, 17), bbox=BoundingBox(Range(0, 100), Range(0, 17)), stride=(1, 100), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(100, 17), size=1700, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=F,	  memoryLayout=DenseMemoryLayout(shape=(17, 45), bbox=BoundingBox(Range(0, 17), Range(0, 45)), stride=(1, 17), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(17, 45), size=765, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 100), Range(0, 45), Range(0, 17))), 'matrix_a': DenseMatrix{name = E, num. rows = 100, num. columns = 17, leading dimension = 100, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 100, 17]}, 'matrix_b': DenseMatrix{name = F, num. rows = 17, num. columns = 45, leading dimension = 17, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 17, 45]}, 'matrix_c': DenseMatrix{name = A, num. rows = 100, num. columns = 45, leading dimension = 100, direction = DataFlowDirection.SINK, bbox = [0, 0, 100, 45]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F', 'A, extraOffset_A', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=A,	  memoryLayout=DenseMemoryLayout(shape=(100, 45), bbox=BoundingBox(Range(0, 100), Range(0, 45)), stride=(1, 100), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(100, 45), size=4500, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=E,	  memoryLayout=DenseMemoryLayout(shape=(100, 17), bbox=BoundingBox(Range(0, 100), Range(0, 17)), stride=(1, 100), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(100, 17), size=1700, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=F,	  memoryLayout=DenseMemoryLayout(shape=(17, 45), bbox=BoundingBox(Range(0, 17), Range(0, 45)), stride=(1, 17), align=<yateto.arch.Architecture object at 0x7f3bda2fff90>),	  eqspp=dense(shape=(17, 45), size=765, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 100), Range(0, 45), Range(0, 17))), 'matrix_a': DenseMatrix{name = E, num. rows = 100, num. columns = 17, leading dimension = 100, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 100, 17]}, 'matrix_b': DenseMatrix{name = F, num. rows = 17, num. columns = 45, leading dimension = 17, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 17, 45]}, 'matrix_c': DenseMatrix{name = A, num. rows = 100, num. columns = 45, leading dimension = 100, direction = DataFlowDirection.SINK, bbox = [0, 0, 100, 45]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F', 'A, extraOffset_A', 'numElements', 'flags', 'streamPtr']})
        const float * const __restrict__ glb_F = &F[batchID][0 + F_extraOffset];
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float * const __restrict__ glb_E = &E[batchID][0 + E_extraOffset];
        float reg0[45] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[2465+4500];
        float * localShrMem0 = &totalShrMem[(2465+4500) * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 13; ++i) {
            shrRegion0[threadIdx.x + i * 128] = glb_E[threadIdx.x + i * 128];
          }
          if (threadIdx.x < 36) {
            shrRegion0[threadIdx.x + 1664] = glb_E[threadIdx.x + 1664];
          }
        }
        
        float* shrRegion1 = &localShrMem0[1700];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 5; ++i) {
            shrRegion1[threadIdx.x + i * 128] = glb_F[threadIdx.x + i * 128];
          }
          if (threadIdx.x < 125) {
            shrRegion1[threadIdx.x + 640] = glb_F[threadIdx.x + 640];
          }
        }

        float* shrRegion2 = &localShrMem0[2465];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 35; ++i) {
            shrRegion2[threadIdx.x + i * 128] = glb_A[threadIdx.x + i * 128];
          }
          if (threadIdx.x < 20) {
            shrRegion2[threadIdx.x + 35*128] = glb_A[threadIdx.x + 35*128];
          }
        }
       
        __syncthreads();
        if (threadIdx.x < 100) {
          float value;
        
          #pragma unroll
          for (int k = 0; k < 17; ++k) {
            value = shrRegion0[threadIdx.x + k * 100];
        
            #pragma unroll
            for (int n = 0; n < 45; ++n) {
              reg0[n] += value * shrRegion1[k + 17 * n];
            }
          }
        }
        if (threadIdx.x < 100) {
          #pragma unroll
          for (int n = 0; n < 45; ++n) {
            glb_A[threadIdx.x + 100 * n] = reg0[n] + shrRegion2[threadIdx.x + 100 * n];
          }
        }
        
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d17_45_d100_45_d100_17__alpha_1_0_beta_1_0_p_p_p__4972d7a(float ** A, int A_extraOffset, const float * const * E, int E_extraOffset, const float * const * F, int F_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(128, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d17_45_d100_45_d100_17__alpha_1_0_beta_1_0_p_p_p__4972d7a<<<grid,block,0,stream>>>(A, A_extraOffset, E, E_extraOffset, F, F_extraOffset, numElements, flags);
  CHECK_ERR;
}


__global__ void 
__launch_bounds__(480)
 kernel_sproduct_NT_NT_NT__d45_d10_10_d10_10_45__alpha_1_0_p_p_p__1e400ba(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      {
        const float * const __restrict__ glb_B = &B[batchID][0 + B_extraOffset];
        const float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float * const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        float reg0[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[145];
        float * localShrMem0 = &totalShrMem[145 * threadIdx.y];

        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedTensorLoader
        {
          if (threadIdx.x < 45) {
            shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
          }
        }

        float* shrRegion1 = &localShrMem0[45];
        // using ExtendedTensorLoader
        {
          if (threadIdx.x < 100) {
            shrRegion1[threadIdx.x + 0] = glb_X[threadIdx.x + 0];
          }
        }
        __syncthreads();
        /*
        This is the product kernel created from the following YaTeTo description:
        Description(
        	alpha: 1.0
        	add: True
        	result: IndexedTensorDescription(name=A, indices=kpm, memoryLayout=DenseMemoryLayout(shape=(10, 10, 45), bbox=BoundingBox(Range(0, 10), Range(0, 10), Range(0, 45)), stride=(1, 10, 100), align=<yateto.arch.Architecture object at 0x7f3bda32f2d0>), eqspp=dense(shape=(10, 10, 45), size=4500, ndim=3), is_compute_constant=False, is_temporary=False)
        	leftTerm: IndexedTensorDescription(name=B, indices=m, memoryLayout=DenseMemoryLayout(shape=(45,), bbox=BoundingBox(Range(0, 45)), stride=(1,), align=<yateto.arch.Architecture object at 0x7f3bda32f2d0>), eqspp=dense(shape=(45,), size=45, ndim=1), is_compute_constant=False, is_temporary=False)
        	rightTerm: IndexedTensorDescription(name=X, indices=kp, memoryLayout=DenseMemoryLayout(shape=(10, 10), bbox=BoundingBox(Range(0, 10), Range(0, 10)), stride=(1, 10), align=<yateto.arch.Architecture object at 0x7f3bda32f2d0>), eqspp=dense(shape=(10, 10), size=100, ndim=2), is_compute_constant=False, is_temporary=False)
        	isACsc: False
        	isBCsc: False
        	loopRanges: {'m': Range(0, 45), 'k': Range(0, 10), 'p': Range(0, 10)}
        )
        */
        if (threadIdx.x < 450) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 10;
          rows_left -= row_offset_1 * 10;
          const int dim_offset_m = row_offset_1;
          const int row_offset_0 = rows_left;
          const int dim_offset_k = row_offset_0;
          #pragma unroll
          for (int p = 0; p < 10; ++p) {
            reg0[p] = shrRegion0[dim_offset_m * 1] * shrRegion1[dim_offset_k * 1 + p * 10];
          }
        }
        if (threadIdx.x < 450) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 10;
          rows_left -= row_offset_1 * 10;
          const int row_offset_0 = rows_left;
          #pragma unroll
          for (int i = 0; i < 10; ++i) {
            glb_A[row_offset_0 * 1 + row_offset_1 * 100 + i * 10] = reg0[i] + 1.0 * glb_A[row_offset_0 * 1 + row_offset_1 * 100 + i * 10];
          }
        }
      }
    }
  }
}
void sproduct_NT_NT_NT__d45_d10_10_d10_10_45__alpha_1_0_p_p_p__1e400ba(float ** A, int A_extraOffset, const float * const * B, int B_extraOffset, const float * const * X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(480, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sproduct_NT_NT_NT__d45_d10_10_d10_10_45__alpha_1_0_p_p_p__1e400ba<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}



int main(){
  constexpr size_t num_els = 116141;
  float* A = new float[4500 * num_els]{0.f};
  float* B = new float[45 * num_els]{0.f};
  float* C = new float[320 * num_els]{0.f};
  float* D = new float[320 * num_els]{0.f};
  float* E = new float[1700 * num_els]{0.f};
  float* F = new float[765 * num_els]{0.f};
  float* X = new float[100 * num_els]{0.f};
  float* R1 = new float[4500 * num_els]{0.f};
  float* R2 = new float[4500 * num_els]{0.f};
  //float* Ri1 = new float[100 * num_els]{0.f};
  //float* Ri2 = new float[4500 * num_els]{0.f};
  //float* Ri1c = new float[100 * num_els]{0.f};
  //float* Ri2c = new float[4500 * num_els]{0.f};


  float* coreA = new float[4500];
  float* coreB = new float[45];
  float* coreC = new float[320];
  float* coreD = new float[320];
  float* coreE = new float[1700];
  float* coreF = new float[765];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < 4500; i++){
    coreA[i] = distribution(gen);
  }
  for (size_t i = 0; i < 45; i++){
    coreB[i] = distribution(gen);
  }
  for (size_t i = 0; i < 320; i++){
    coreC[i] = distribution(gen);
  }
  for (size_t i = 0; i < 320; i++){
    coreD[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1700; i++){
    coreE[i] = distribution(gen);
  }
  for (size_t i = 0; i < 765; i++){
    coreF[i] = distribution(gen);
  }

  for (size_t i = 0; i < num_els; i++){
      std::memcpy(&A[i * 4500], &coreA[0], 4500 * sizeof(float));
      std::memcpy(&B[i * 45], &coreB[0], 45 * sizeof(float));
      std::memcpy(&C[i * 320], &coreC[0], 320 * sizeof(float));
      std::memcpy(&D[i * 320], &coreD[0], 320 * sizeof(float));
      std::memcpy(&E[i * 1700], &coreE[0], 1700 * sizeof(float));
      std::memcpy(&F[i * 765], &coreF[0], 765 * sizeof(float));
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

  cudaMalloc((void **)&A_dev, sizeof(float) * 4500 * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * 45 * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * 320 * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * 320 * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * 1700 * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * 765 * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * 100 * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
 
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4500 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 45 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 320 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * 320 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * 1700 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * 765 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 100 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){
    A_dev_begins[i] = A_dev + i * 4500;
    B_dev_begins[i] = B_dev + i * 45;
    C_dev_begins[i] = C_dev + i * 320;
    D_dev_begins[i] = D_dev + i * 320;
    E_dev_begins[i] = E_dev + i * 1700;
    F_dev_begins[i] = F_dev + i * 765;
    X_dev_begins[i] = X_dev + i * 100;
  }

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  sloopOverGEMM_NT_NT_NT__d10_32_d32_10_d10_10__alpha_1_0_beta_0_0_p_p_p__6a7ea95(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 100 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Dimensions: " << 10 << ", " << 17 << ", " << 45 << ", " << 10 << ", " << 32 << std::endl;

  float elapsedTimeT1 = 0.0;
  float elapsedTimeT2 = 0.0;
  float elapsedTimeT3 = 0.0; 
  cudaEvent_t startT1, stopT1;
  cudaEvent_t startT2, stopT2;
  cudaEvent_t startT3, stopT3;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  sloopOverGEMM_NT_NT_NT__d10_32_d32_10_d10_10__alpha_1_0_beta_0_0_p_p_p__6a7ea95(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT1, startT1, stopT1); CHECK_ERR;
  //cudaDeviceSynchronize(); CHECK_ERR;

  //cudaMemcpy(Ri1, X_dev, sizeof(float) * 100 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventRecord(startT2); CHECK_ERR;
  sloopOverGEMM_NT_NT_NT__d17_45_d100_45_d100_17__alpha_1_0_beta_1_0_p_p_p__4972d7a(A_dev_begins_dev, 0, E_dev_begins_dev, 0, F_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT2); CHECK_ERR;
  cudaEventSynchronize(stopT2); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;
  //cudaDeviceSynchronize(); CHECK_ERR;

  //cudaMemcpy(Ri2, A_dev, sizeof(float) * 4500 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

  cudaEventCreate(&startT3); CHECK_ERR;
  cudaEventCreate(&stopT3); CHECK_ERR;
  cudaEventRecord(startT3); CHECK_ERR;
  sproduct_NT_NT_NT__d45_d10_10_d10_10_45__alpha_1_0_p_p_p__1e400ba(A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT3); CHECK_ERR;
  cudaEventSynchronize(stopT3); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3); CHECK_ERR;
  double elapsedTime = elapsedTimeT1 + elapsedTimeT2 + elapsedTimeT3;
  cudaDeviceSynchronize(); CHECK_ERR;
  
  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms" << std::endl; 
  cudaMemcpy(R1, A_dev, sizeof(float) * 4500 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4500 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  double fp_per_el = 172900;
  double ls_per_el = 48600;
  double fp_unfused_per_el = 172900;
  double ls_unfused_per_el = 85400;
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  fp_unfused_per_el *= num_els;
  ls_unfused_per_el *= num_els;
  std::cout << "Gemmforge Theoretical Fused Kernel GFLOPs/s: " << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational Theoretical Fused intensity: " << fp_per_el / ls_per_el << std::endl;
  std::cout << "Gemmforge GFLOPs/s: " << fp_unfused_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational intensity: " << fp_unfused_per_el / ls_unfused_per_el << std::endl;
  double peakFLOPGiven = 29767.7;
  double peakBandwidthGiven = 760.08;

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with Gemmforge" << std::endl;
    //std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;
    double obtainable_unfused_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_unfused_per_el) / static_cast<double>(ls_unfused_per_el)));
    std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTime) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with Gemmforge" << std::endl;
    //std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTime) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with cuTensor" << std::endl;
    double obtainable_unfused_peak_k1 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(6400) / static_cast<double>(2960)));
    std::cout << 100.0*(6400 * num_els  * 1e-6 / elapsedTimeT1) / obtainable_unfused_peak_k1 << " % of roof w. respect to Kernel1 intensity achieved with Gemmforge" << std::endl;
    double obtainable_unfused_peak_k2 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(157500) / static_cast<double>(45860)));
    std::cout << 100.0*(157500 * num_els  * 1e-6 / elapsedTimeT2) / obtainable_unfused_peak_k2 << " % of roof w. respect to Kernel2 intensity achieved with Gemmforge" << std::endl;
    double obtainable_unfused_peak_k3 = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(9000) / static_cast<double>(36580)));
    std::cout << 100.0*(9000 * num_els * 1e-6 / elapsedTimeT3) / obtainable_unfused_peak_k3 << " % of roof w. respect to Kernel3 intensity achieved with Gemmforge" << std::endl;
  }

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4500 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 100 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  if constexpr (!false){
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
  {
    float alphaK1 = 1.0f;
    float betaK1 = 0.0f;
    float alphaK2 = 1.0f;
    float betaK2 = 1.0;
    float alphaK3 = 1.0f;
    float betaK3 = 1.0;

    std::vector<int> modeA{'k', 'p', 'm', 'b'};
    std::vector<int> modeB{'m', 'b'};
    std::vector<int> modeC{'k', 'q', 'b'};
    std::vector<int> modeD{'q', 'p', 'b'};
    std::vector<int> modeE{'k', 'p', 'l', 'b'};
    std::vector<int> modeF{'l', 'm', 'b'};
    std::vector<int> modeX{'k', 'p', 'b'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();
    int nmodeD = modeD.size();
    int nmodeE = modeE.size();
    int nmodeF = modeF.size();
    int nmodeX = modeX.size();

    std::unordered_map<int, int64_t> extent;
    // Derived from the kernel
    extent['k'] = 10;
    extent['l'] = 17;
    extent['m'] = 45;
    extent['p'] = 10;
    extent['q'] = 32;
    extent['b'] = num_els;

    std::vector<int64_t> extentA;
    for (auto mode : modeA) {
        extentA.push_back(extent[mode]);
    }
    std::vector<int64_t> extentB;
    for (auto mode : modeB) {
        extentB.push_back(extent[mode]);
    }
    std::vector<int64_t> extentC;
    for (auto mode : modeC) {
        extentC.push_back(extent[mode]);
    }
    std::vector<int64_t> extentD;
    for (auto mode : modeD) {
        extentD.push_back(extent[mode]);
    }
    std::vector<int64_t> extentE;
    for (auto mode : modeE) {
        extentE.push_back(extent[mode]);
    }
    std::vector<int64_t> extentF;
    for (auto mode : modeF) {
        extentF.push_back(extent[mode]);
    }
    std::vector<int64_t> extentX;
    for (auto mode : modeX) {
        extentX.push_back(extent[mode]);
    }
    
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
    {
        if (cudaSuccess != cudaMalloc(&work, maxWorkSize))
        {
            work = nullptr;
            maxWorkSize = 0;
            worksize1 = 0;
            worksize2 = 0;
            worksize3 = 0;
            cudaGetLastError(); // Clear last error to save CHECK_ERR;
        } else {
            worksize1 = maxWorkSize;
            worksize2 = maxWorkSize;
            worksize3 = maxWorkSize;
        }
    }


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
    //cudaMemcpy(Ri1c, X_dev, sizeof(float) * 100 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

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
    //cudaMemcpy(Ri2c, A_dev, sizeof(float) * 4500 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

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
    
    cudaMemcpy(R2, A_dev, sizeof(float) * 4500 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;

    cudaFree(work);
  }

  float elapsedTimeCuTensor = elapsedTimeCT1 + elapsedTimeCT2 + elapsedTimeCT2;
  if (peakFLOPGiven > 0.1 && peakBandwidthGiven){
    double obtainable_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_per_el) / static_cast<double>(ls_per_el)));
    std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTimeCuTensor) / obtainable_peak << " % of roof w. respect to operational intensity achieved with cuTensor" << std::endl;

    double obtainable_unfused_peak = std::min(static_cast<double>(peakFLOPGiven), static_cast<double>(peakBandwidthGiven * static_cast<double>(fp_unfused_per_el) / static_cast<double>(ls_unfused_per_el)));
    std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTimeCuTensor) / obtainable_unfused_peak << " % of roof w. respect to unfused operational intensity achieved with cuTensor" << std::endl;
  }

  /*
  bool i1results_wrong = false;
  for (size_t i = 0; i < 100 * num_els; i++){
    if (std::abs(Ri1[i] - Ri1c[i]) > 1.0f) {
      std::cout << "Intermediate Results 1 do not match, problem first at offset " << i << " :_(" << std::endl;
      i1results_wrong = true;
      break;
    }
  }
  if (!i1results_wrong){
    std::cout << "Gemmforge and cuTensor contraction intermediate results 1 match! :)" << std::endl;
  }
  
  bool i2results_wrong = false;
  for (size_t i = 0; i < 4500 * num_els; i++){
    if (std::abs(Ri2[i] - Ri2c[i]) > 1.0f) {
      std::cout << "Intermediate Results 2 do not match, problem first at offset " << i << " :_(" << std::endl;
      i2results_wrong = true;
      break;
    }
  }
  if (!i2results_wrong){
    std::cout << "Gemmforge and cuTensor contraction intermediate results 2 match! :)" << std::endl;
  }
  */

  bool results_wrong = false;
  for (size_t i = 0; i < 4500 * num_els; i++){
    if (std::abs(R1[i] - R2[i]) > 5.0f) {
      std::cout << "Results do not match, problem first at offset " << i << " :_(" << std::endl;
      results_wrong = true;
      break;
    }
  }
  if (!results_wrong){
    std::cout << "Gemmforge and cuTensor contraction results match! :)" << std::endl;
  }
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

