
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      std::cout << "Error: " << cutensorGetErrorString(err) << std::endl;      \
      std::cout << __FILE__ << " " << __LINE__ << std::endl;                   \
    }                                                                          \
  }

#define CHECK_ERR checkErr(__FILE__, __LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
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
    std::cout << std::endl
              << File << ", line " << Line << ": " << cudaGetErrorString(Error)
              << " (" << Error << ")" << std::endl;

    if (PrevLine > 0)
      std::cout << "Previous CUDA call:" << std::endl
                << PrevFile << ", line " << PrevLine << std::endl;
  }
  PrevFile = File;
  PrevLine = Line;
#endif
}

__global__ void __launch_bounds__(32)
    kernel_sloopOverGEMM_NT_NT_NT__d21_61_d21_22_d61_22__alpha_1_0_beta_0_0_p_p_p__7610af5(
        const float *const *C, int C_extraOffset, const float *const *D,
        int D_extraOffset, float **X, int X_extraOffset, unsigned numElements,
        unsigned *flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=X,
      memoryLayout=DenseMemoryLayout(shape=(21, 22), bbox=BoundingBox(Range(0,
      21), Range(0, 22)), stride=(1, 21), align=<yateto.arch.Architecture object
      at 0x7f0cf15817d0>),	  eqspp=dense(shape=(21, 22), size=462, ndim=2),
      is_compute_constant=False,	  is_temporary=False),
      leftTerm=TensorDescription(  name=C,
      memoryLayout=DenseMemoryLayout(shape=(21, 61), bbox=BoundingBox(Range(0,
      21), Range(0, 61)), stride=(1, 21), align=<yateto.arch.Architecture object
      at 0x7f0cf15817d0>),	  eqspp=dense(shape=(21, 61), size=1281,
      ndim=2), is_compute_constant=False,	  is_temporary=False),
      rightTerm=TensorDescription(  name=D,
      memoryLayout=DenseMemoryLayout(shape=(61, 22), bbox=BoundingBox(Range(0,
      61), Range(0, 22)), stride=(1, 61), align=<yateto.arch.Architecture object
      at 0x7f0cf15817d0>),	  eqspp=dense(shape=(61, 22), size=1342,
      ndim=2), is_compute_constant=False,	  is_temporary=False),
      transA=False, transB=False,	  alpha=1.0,	  beta=0.0,
      prefetchName=None, isACsc=False,	  isBCsc=False, alignedA=False,
      alignedC=False, mnk=(Range(0, 21), Range(0, 22), Range(0, 61))),
      'matrix_a': DenseMatrix{name = C, num. rows = 21, num. columns = 61,
      leading dimension = 21, direction = DataFlowDirection.SOURCE, bbox = [0,
      0, 21, 61]}, 'matrix_b': DenseMatrix{name = D, num. rows = 61, num.
      columns = 22, leading dimension = 61, direction =
      DataFlowDirection.SOURCE, bbox = [0, 0, 61, 22]}, 'matrix_c':
      DenseMatrix{name = X, num. rows = 21, num. columns = 22, leading dimension
      = 21, direction = DataFlowDirection.SINK, bbox = [0, 0, 21, 22]}, 'args':
      ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X',
      'numElements', 'flags', 'streamPtr']})
      */
      {
        //('gemm', {'descr': Description(  result=TensorDescription(  name=X,
        // memoryLayout=DenseMemoryLayout(shape=(21, 22),
        // bbox=BoundingBox(Range(0, 21), Range(0, 22)), stride=(1, 21),
        // align=<yateto.arch.Architecture object at 0x7f0cf15817d0>),
        // eqspp=dense(shape=(21, 22), size=462, ndim=2),
        // is_compute_constant=False,	  is_temporary=False),
        // leftTerm=TensorDescription(  name=C,
        // memoryLayout=DenseMemoryLayout(shape=(21, 61),
        // bbox=BoundingBox(Range(0, 21), Range(0, 61)), stride=(1, 21),
        // align=<yateto.arch.Architecture object at 0x7f0cf15817d0>),
        // eqspp=dense(shape=(21, 61), size=1281, ndim=2),
        // is_compute_constant=False,	  is_temporary=False),
        // rightTerm=TensorDescription(  name=D,
        // memoryLayout=DenseMemoryLayout(shape=(61, 22),
        // bbox=BoundingBox(Range(0, 61), Range(0, 22)), stride=(1, 61),
        // align=<yateto.arch.Architecture object at 0x7f0cf15817d0>),
        // eqspp=dense(shape=(61, 22), size=1342, ndim=2),
        // is_compute_constant=False,	  is_temporary=False), transA=False,
        // transB=False,	  alpha=1.0,	  beta=0.0, prefetchName=None,
        // isACsc=False,	  isBCsc=False,	  alignedA=False,
        // alignedC=False, mnk=(Range(0, 21), Range(0, 22), Range(0, 61))),
        // 'matrix_a': DenseMatrix{name = C, num. rows = 21, num. columns = 61,
        // leading dimension = 21, direction = DataFlowDirection.SOURCE, bbox =
        // [0, 0, 21, 61]}, 'matrix_b': DenseMatrix{name = D, num. rows = 61,
        // num. columns = 22, leading dimension = 61, direction =
        // DataFlowDirection.SOURCE, bbox = [0, 0, 61, 22]}, 'matrix_c':
        // DenseMatrix{name = X, num. rows = 21, num. columns = 22, leading
        // dimension = 21, direction = DataFlowDirection.SINK, bbox = [0, 0, 21,
        // 22]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X,
        // extraOffset_X', 'numElements', 'flags', 'streamPtr']})
        const float *const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        float *const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        const float *const __restrict__ glb_D = &D[batchID][0 + D_extraOffset];
        float reg0[21] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__ __align__(16) float totalShrMem1[1952];
        float *localShrMem0 = &totalShrMem1[1952 * threadIdx.y];
        __shared__ __align__(16) float totalShrMem2[1408];
        float *localShrMem1 = &totalShrMem2[1408 * threadIdx.y];
        __shared__ __align__(16) float shrC[22 * 21];

        float *shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
#pragma unroll
          for (int i = 0; i < 61; ++i) {
            if (threadIdx.x < 21) {
              shrRegion0[threadIdx.x + i * 32] = glb_C[threadIdx.x + i * 21];
            }
          }
        }

        float *shrRegion1 = &localShrMem1[0];
        // using ExtendedPatchLoader
        {
#pragma unroll
          for (int i = 0; i < 22; ++i) {
            shrRegion1[threadIdx.x + i * 64] = glb_D[threadIdx.x + i * 61];
            if (threadIdx.x < 29) {
              shrRegion1[threadIdx.x + 32 + i * 64] =
                  glb_D[threadIdx.x + 32 + i * 61];
            }
          }
        }
        __syncwarp();
        if (threadIdx.x < 22) {

          // float value;

          
          float4 values;
          #pragma unroll
          for (int k = 0; k < 60; k += 4) {
            values.x = shrRegion1[threadIdx.x * 64 + 0 + k];
            values.y = shrRegion1[threadIdx.x * 64 + 1 + k];
            values.z = shrRegion1[threadIdx.x * 64 + 2 + k];
            values.w = shrRegion1[threadIdx.x * 64 + 3 + k];

          #pragma unroll
            for (int n = 0; n < 20; n += 4) {
              reg0[n + 0] += values.x * shrRegion0[k*32 + 0 + n];
              reg0[n + 1] += values.x * shrRegion0[k*32 + 1 + n];
              reg0[n + 2] += values.x * shrRegion0[k*32 + 2 + n];
              reg0[n + 3] += values.x * shrRegion0[k*32 + 3 + n];
              reg0[n + 0] += values.y * shrRegion0[(k+1)*32 + 0 + n];
              reg0[n + 1] += values.y * shrRegion0[(k+1)*32 + 1 + n];
              reg0[n + 2] += values.y * shrRegion0[(k+1)*32 + 2 + n];
              reg0[n + 3] += values.y * shrRegion0[(k+1)*32 + 3 + n];
              reg0[n + 0] += values.z * shrRegion0[(k+2)*32 + 0 + n];
              reg0[n + 1] += values.z * shrRegion0[(k+2)*32 + 1 + n];
              reg0[n + 2] += values.z * shrRegion0[(k+2)*32 + 2 + n];
              reg0[n + 3] += values.z * shrRegion0[(k+2)*32 + 3 + n];
              reg0[n + 0] += values.w * shrRegion0[(k+3)*32 + 0 + n];
              reg0[n + 1] += values.w * shrRegion0[(k+3)*32 + 1 + n];
              reg0[n + 2] += values.w * shrRegion0[(k+3)*32 + 2 + n];
              reg0[n + 3] += values.w * shrRegion0[(k+3)*32 + 3 + n];
            }
            reg0[20] += values.x * shrRegion0[20 + k*32];
            reg0[20] += values.y * shrRegion0[20 + (k+1)*32];
            reg0[20] += values.z * shrRegion0[20 + (k+2)*32];
            reg0[20] += values.w * shrRegion0[20 + (k+3)*32];
          }
          values.x = shrRegion1[threadIdx.x * 64 + 0 + 60];
          #pragma unroll
          for (int n = 0; n < 21; n += 1) {
            reg0[n] += values.x * shrRegion0[n + 60*32];
          }
          /*
          float value;
#pragma unroll
          for (int k = 0; k < 61; k++) {
            value = shrRegion1[threadIdx.x * 64 + k];
#pragma unroll
            for (int m = 0; m < 21; m++) {
              reg0[m] += value * shrRegion0[m + k * 32];
            }
          }
          */
        }

        if (threadIdx.x < 22) {
#pragma unroll
          for (int m = 0; m < 21; ++m) {
            shrC[threadIdx.x * 21 + m] = reg0[m];
          }
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 14; i++) {
          glb_X[threadIdx.x + i * 32] = shrC[threadIdx.x + i * 32];
        }
        if (threadIdx.x < 14) {
          glb_X[threadIdx.x + 14 * 32] = shrC[threadIdx.x + 14 * 32];
        }
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d21_61_d21_22_d61_22__alpha_1_0_beta_0_0_p_p_p__7610af5(
    const float *const *C, int C_extraOffset, const float *const *D,
    int D_extraOffset, float **X, int X_extraOffset, unsigned numElements,
    unsigned *flags, void *streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream =
      (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d21_61_d21_22_d61_22__alpha_1_0_beta_0_0_p_p_p__7610af5<<<
      grid, block, 0, stream>>>(C, C_extraOffset, D, D_extraOffset, X,
                                X_extraOffset, numElements, flags);
  CHECK_ERR;
}

__global__ void __launch_bounds__(480)
    kernel_sloopOverGEMM_NT_NT_NT__d462_13_d462_10_d13_10__alpha_1_0_beta_1_0_p_p_p__419e9f1(
        float **A, int A_extraOffset, const float *const *E, int E_extraOffset,
        const float *const *F, int F_extraOffset, unsigned numElements,
        unsigned *flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=A,
      memoryLayout=DenseMemoryLayout(shape=(462, 10), bbox=BoundingBox(Range(0,
      462), Range(0, 10)), stride=(1, 462), align=<yateto.arch.Architecture
      object at 0x7f0cfd758d90>),	  eqspp=dense(shape=(462, 10),
      size=4620, ndim=2),	  is_compute_constant=False,
      is_temporary=False), leftTerm=TensorDescription(  name=E,
      memoryLayout=DenseMemoryLayout(shape=(462, 13), bbox=BoundingBox(Range(0,
      462), Range(0, 13)), stride=(1, 462), align=<yateto.arch.Architecture
      object at 0x7f0cfd758d90>),	  eqspp=dense(shape=(462, 13),
      size=6006, ndim=2),	  is_compute_constant=False,
      is_temporary=False), rightTerm=TensorDescription(  name=F,
      memoryLayout=DenseMemoryLayout(shape=(13, 10), bbox=BoundingBox(Range(0,
      13), Range(0, 10)), stride=(1, 13), align=<yateto.arch.Architecture object
      at 0x7f0cfd758d90>),	  eqspp=dense(shape=(13, 10), size=130, ndim=2),
      is_compute_constant=False,	  is_temporary=False),	  transA=False,
      transB=False,	  alpha=1.0,	  beta=1.0,	  prefetchName=None,
      isACsc=False,	  isBCsc=False,	  alignedA=False, alignedC=False,
      mnk=(Range(0, 462), Range(0, 10), Range(0, 13))), 'matrix_a':
      DenseMatrix{name = E, num. rows = 462, num. columns = 13, leading
      dimension = 462, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 462,
      13]}, 'matrix_b': DenseMatrix{name = F, num. rows = 13, num. columns = 10,
      leading dimension = 13, direction = DataFlowDirection.SOURCE, bbox = [0,
      0, 13, 10]}, 'matrix_c': DenseMatrix{name = A, num. rows = 462, num.
      columns = 10, leading dimension = 462, direction = DataFlowDirection.SINK,
      bbox = [0, 0, 462, 10]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F',
      'A, extraOffset_A', 'numElements', 'flags', 'streamPtr']})
      */
      {
        //('gemm', {'descr': Description(  result=TensorDescription(  name=A,
        // memoryLayout=DenseMemoryLayout(shape=(462, 10),
        // bbox=BoundingBox(Range(0, 462), Range(0, 10)), stride=(1, 462),
        // align=<yateto.arch.Architecture object at 0x7f0cfd758d90>),
        // eqspp=dense(shape=(462, 10), size=4620, ndim=2),
        // is_compute_constant=False,	  is_temporary=False),
        // leftTerm=TensorDescription(  name=E,
        // memoryLayout=DenseMemoryLayout(shape=(462, 13),
        // bbox=BoundingBox(Range(0, 462), Range(0, 13)), stride=(1, 462),
        // align=<yateto.arch.Architecture object at 0x7f0cfd758d90>),
        // eqspp=dense(shape=(462, 13), size=6006, ndim=2),
        // is_compute_constant=False,	  is_temporary=False),
        // rightTerm=TensorDescription(  name=F,
        // memoryLayout=DenseMemoryLayout(shape=(13, 10),
        // bbox=BoundingBox(Range(0, 13), Range(0, 10)), stride=(1, 13),
        // align=<yateto.arch.Architecture object at 0x7f0cfd758d90>),
        // eqspp=dense(shape=(13, 10), size=130, ndim=2),
        // is_compute_constant=False,	  is_temporary=False), transA=False,
        // transB=False,	  alpha=1.0,	  beta=1.0, prefetchName=None,
        // isACsc=False,	  isBCsc=False,	  alignedA=False,
        // alignedC=False, mnk=(Range(0, 462), Range(0, 10), Range(0, 13))),
        // 'matrix_a': DenseMatrix{name = E, num. rows = 462, num. columns = 13,
        // leading dimension = 462, direction = DataFlowDirection.SOURCE, bbox =
        // [0, 0, 462, 13]}, 'matrix_b': DenseMatrix{name = F, num. rows = 13,
        // num. columns = 10, leading dimension = 13, direction =
        // DataFlowDirection.SOURCE, bbox = [0, 0, 13, 10]}, 'matrix_c':
        // DenseMatrix{name = A, num. rows = 462, num. columns = 10, leading
        // dimension = 462, direction = DataFlowDirection.SINK, bbox = [0, 0,
        // 462, 10]}, 'args': ['E, extraOffset_E', 'F, extraOffset_F', 'A,
        // extraOffset_A', 'numElements', 'flags', 'streamPtr']})
        const float *const __restrict__ glb_E = &E[batchID][0 + E_extraOffset];
        float *const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float *const __restrict__ glb_F = &F[batchID][0 + F_extraOffset];
        float reg0[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__ __align__(8) float totalShrMem[6136];
        float *localShrMem0 = &totalShrMem[6136 * threadIdx.y];

        float *shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
#pragma unroll
          for (int i = 0; i < 12; ++i) {
            shrRegion0[threadIdx.x + i * 480] = glb_E[threadIdx.x + i * 480];
          }
          if (threadIdx.x < 246) {
            shrRegion0[threadIdx.x + 5760] = glb_E[threadIdx.x + 5760];
          }
        }

        float *shrRegion1 = &localShrMem0[6006];
        // using ExtendedPatchLoader
        {
          if (threadIdx.x < 130) {
            shrRegion1[threadIdx.x + 0] = glb_F[threadIdx.x + 0];
          }
        }
        __syncthreads();
        if (threadIdx.x < 462) {
          float value;

#pragma unroll
          for (int k = 0; k < 13; ++k) {
            value = shrRegion0[threadIdx.x + k * 462];

#pragma unroll
            for (int n = 0; n < 10; ++n) {
              reg0[n] += value * shrRegion1[k + 13 * n];
            }
          }
        }
        if (threadIdx.x < 462) {
#pragma unroll
          for (int n = 0; n < 10; ++n) {
            glb_A[threadIdx.x + 462 * n] =
                reg0[n] + glb_A[threadIdx.x + 462 * n];
          }
        }
      }
    }
  }
}

void sloopOverGEMM_NT_NT_NT__d462_13_d462_10_d13_10__alpha_1_0_beta_1_0_p_p_p__419e9f1(
    float **A, int A_extraOffset, const float *const *E, int E_extraOffset,
    const float *const *F, int F_extraOffset, unsigned numElements,
    unsigned *flags, void *streamPtr) {
  dim3 block(480, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream =
      (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d462_13_d462_10_d13_10__alpha_1_0_beta_1_0_p_p_p__419e9f1<<<
      grid, block, 0, stream>>>(A, A_extraOffset, E, E_extraOffset, F,
                                F_extraOffset, numElements, flags);
  CHECK_ERR;
}

__global__ void __launch_bounds__(224)
    kernel_sproduct_NT_NT_NT__d10_d21_22_10_d21_22__alpha_1_0_p_p_p__9329d0c(
        float **A, int A_extraOffset, const float *const *B, int B_extraOffset,
        const float *const *X, int X_extraOffset, unsigned numElements,
        unsigned *flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      {
        const float *const __restrict__ glb_B = &B[batchID][0 + B_extraOffset];
        float *const __restrict__ glb_A = &A[batchID][0 + A_extraOffset];
        const float *const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        float reg0[22] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__ __align__(8) float totalShrMem[472];
        float *localShrMem0 = &totalShrMem[472 * threadIdx.y];

        float *shrRegion0 = &localShrMem0[0];
        // using ExtendedTensorLoader
        {
          if (threadIdx.x < 10) {
            shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
          }
        }

        float *shrRegion1 = &localShrMem0[10];
        // using ExtendedTensorLoader
        {
          shrRegion1[threadIdx.x + 0] = glb_X[threadIdx.x + 0];
          shrRegion1[threadIdx.x + 224] = glb_X[threadIdx.x + 224];
          if (threadIdx.x < 14) {
            shrRegion1[threadIdx.x + 448] = glb_X[threadIdx.x + 448];
          }
        }
        __syncthreads();
        /*
        This is the product kernel created from the following YaTeTo
        description: Description( alpha: 1.0 add: True result:
        IndexedTensorDescription(name=A, indices=kpm,
        memoryLayout=DenseMemoryLayout(shape=(21, 22, 10),
        bbox=BoundingBox(Range(0, 21), Range(0, 22), Range(0, 10)), stride=(1,
        21, 462), align=<yateto.arch.Architecture object at 0x7f0cf15675d0>),
        eqspp=dense(shape=(21, 22, 10), size=4620, ndim=3),
        is_compute_constant=False, is_temporary=False) leftTerm:
        IndexedTensorDescription(name=B, indices=m,
        memoryLayout=DenseMemoryLayout(shape=(10,), bbox=BoundingBox(Range(0,
        10)), stride=(1,), align=<yateto.arch.Architecture object at
        0x7f0cf15675d0>), eqspp=dense(shape=(10,), size=10, ndim=1),
        is_compute_constant=False, is_temporary=False) rightTerm:
        IndexedTensorDescription(name=X, indices=kp,
        memoryLayout=DenseMemoryLayout(shape=(21, 22), bbox=BoundingBox(Range(0,
        21), Range(0, 22)), stride=(1, 21), align=<yateto.arch.Architecture
        object at 0x7f0cf15675d0>), eqspp=dense(shape=(21, 22), size=462,
        ndim=2), is_compute_constant=False, is_temporary=False) isACsc: False
                isBCsc: False
                loopRanges: {'m': Range(0, 10), 'k': Range(0, 21), 'p': Range(0,
        22)}
        )
        */
        if (threadIdx.x < 210) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 21;
          rows_left -= row_offset_1 * 21;
          const int dim_offset_m = row_offset_1;
          const int row_offset_0 = rows_left;
          const int dim_offset_k = row_offset_0;
#pragma unroll
          for (int p = 0; p < 22; ++p) {
            reg0[p] = shrRegion0[dim_offset_m * 1] *
                      shrRegion1[dim_offset_k * 1 + p * 21];
          }
        }
        if (threadIdx.x < 210) {
          int rows_left = threadIdx.x;
          const int row_offset_1 = rows_left / 21;
          rows_left -= row_offset_1 * 21;
          const int row_offset_0 = rows_left;
#pragma unroll
          for (int i = 0; i < 22; ++i) {
            glb_A[row_offset_0 * 1 + row_offset_1 * 462 + i * 21] =
                reg0[i] +
                1.0 * glb_A[row_offset_0 * 1 + row_offset_1 * 462 + i * 21];
          }
        }
      }
    }
  }
}
void sproduct_NT_NT_NT__d10_d21_22_10_d21_22__alpha_1_0_p_p_p__9329d0c(
    float **A, int A_extraOffset, const float *const *B, int B_extraOffset,
    const float *const *X, int X_extraOffset, unsigned numElements,
    unsigned *flags, void *streamPtr) {
  dim3 block(224, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream =
      (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sproduct_NT_NT_NT__d10_d21_22_10_d21_22__alpha_1_0_p_p_p__9329d0c<<<
      grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, X,
                                X_extraOffset, numElements, flags);
  CHECK_ERR;
}

int main() {
  constexpr size_t num_els = 65053;
  float *A = new float[4620 * num_els]{0.f};
  float *B = new float[10 * num_els]{0.f};
  float *C = new float[1281 * num_els]{0.f};
  float *D = new float[1342 * num_els]{0.f};
  float *E = new float[6006 * num_els]{0.f};
  float *F = new float[130 * num_els]{0.f};
  float *X = new float[462 * num_els]{0.f};
  float *R1 = new float[4620 * num_els]{0.f};
  float *R2 = new float[4620 * num_els]{0.f};
  // float* Ri1 = new float[462 * num_els]{0.f};
  // float* Ri2 = new float[4620 * num_els]{0.f};
  // float* Ri1c = new float[462 * num_els]{0.f};
  // float* Ri2c = new float[4620 * num_els]{0.f};

  float *coreA = new float[4620];
  float *coreB = new float[10];
  float *coreC = new float[1281];
  float *coreD = new float[1342];
  float *coreE = new float[6006];
  float *coreF = new float[130];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < 4620; i++) {
    coreA[i] = distribution(gen);
  }
  for (size_t i = 0; i < 10; i++) {
    coreB[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1281; i++) {
    coreC[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1342; i++) {
    coreD[i] = distribution(gen);
  }
  for (size_t i = 0; i < 6006; i++) {
    coreE[i] = distribution(gen);
  }
  for (size_t i = 0; i < 130; i++) {
    coreF[i] = distribution(gen);
  }

  for (size_t i = 0; i < num_els; i++) {
    std::memcpy(&A[i * 4620], &coreA[0], 4620 * sizeof(float));
    std::memcpy(&B[i * 10], &coreB[0], 10 * sizeof(float));
    std::memcpy(&C[i * 1281], &coreC[0], 1281 * sizeof(float));
    std::memcpy(&D[i * 1342], &coreD[0], 1342 * sizeof(float));
    std::memcpy(&E[i * 6006], &coreE[0], 6006 * sizeof(float));
    std::memcpy(&F[i * 130], &coreF[0], 130 * sizeof(float));
  }

  float *A_dev = nullptr;
  float *B_dev = nullptr;
  float *C_dev = nullptr;
  float *D_dev = nullptr;
  float *E_dev = nullptr;
  float *F_dev = nullptr;
  float *X_dev = nullptr;

  float **A_dev_begins = new float *[num_els];
  float **B_dev_begins = new float *[num_els];
  float **C_dev_begins = new float *[num_els];
  float **D_dev_begins = new float *[num_els];
  float **E_dev_begins = new float *[num_els];
  float **F_dev_begins = new float *[num_els];
  float **X_dev_begins = new float *[num_els];

  float **A_dev_begins_dev = nullptr;
  float **B_dev_begins_dev = nullptr;
  float **C_dev_begins_dev = nullptr;
  float **D_dev_begins_dev = nullptr;
  float **E_dev_begins_dev = nullptr;
  float **F_dev_begins_dev = nullptr;
  float **X_dev_begins_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * 4620 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * 10 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * 1281 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * 1342 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * 6006 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * 130 * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * 462 * num_els);
  CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float *) * num_els);
  CHECK_ERR;

  cudaDeviceSynchronize();
  CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4620 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 10 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 1281 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * 1342 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * 6006 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * 130 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 462 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  for (size_t i = 0; i < num_els; i++) {
    A_dev_begins[i] = A_dev + i * 4620;
    B_dev_begins[i] = B_dev + i * 10;
    C_dev_begins[i] = C_dev + i * 1281;
    D_dev_begins[i] = D_dev + i * 1342;
    E_dev_begins[i] = E_dev + i * 6006;
    F_dev_begins[i] = F_dev + i * 130;
    X_dev_begins[i] = X_dev + i * 462;
  }

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins,
             sizeof(float *) * num_els, cudaMemcpyHostToDevice);
  CHECK_ERR;

  sloopOverGEMM_NT_NT_NT__d21_61_d21_22_d61_22__alpha_1_0_beta_0_0_p_p_p__7610af5(
      C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els,
      nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 462 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  std::cout << "Dimensions: " << 21 << ", " << 13 << ", " << 10 << ", " << 22
            << ", " << 61 << std::endl;

  float elapsedTimeT1 = 0.0;
  float elapsedTimeT2 = 0.0;
  float elapsedTimeT3 = 0.0;
  cudaEvent_t startT1, stopT1;
  cudaEvent_t startT2, stopT2;
  cudaEvent_t startT3, stopT3;
  cudaEventCreate(&startT1);
  CHECK_ERR;
  cudaEventCreate(&stopT1);
  CHECK_ERR;
  cudaEventRecord(startT1);
  CHECK_ERR;
  sloopOverGEMM_NT_NT_NT__d21_61_d21_22_d61_22__alpha_1_0_beta_0_0_p_p_p__7610af5(
      C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els,
      nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopT1);
  CHECK_ERR;
  cudaEventSynchronize(stopT1);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT1, startT1, stopT1);
  CHECK_ERR;
  // cudaDeviceSynchronize(); CHECK_ERR;

  // cudaMemcpy(Ri1, X_dev, sizeof(float) * 462 * num_els,
  // cudaMemcpyDeviceToHost); CHECK_ERR;

  cudaEventCreate(&startT2);
  CHECK_ERR;
  cudaEventCreate(&stopT2);
  CHECK_ERR;
  cudaEventRecord(startT2);
  CHECK_ERR;
  sloopOverGEMM_NT_NT_NT__d462_13_d462_10_d13_10__alpha_1_0_beta_1_0_p_p_p__419e9f1(
      A_dev_begins_dev, 0, E_dev_begins_dev, 0, F_dev_begins_dev, 0, num_els,
      nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopT2);
  CHECK_ERR;
  cudaEventSynchronize(stopT2);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2);
  CHECK_ERR;
  // cudaDeviceSynchronize(); CHECK_ERR;

  // cudaMemcpy(Ri2, A_dev, sizeof(float) * 4620 * num_els,
  // cudaMemcpyDeviceToHost); CHECK_ERR;

  cudaEventCreate(&startT3);
  CHECK_ERR;
  cudaEventCreate(&stopT3);
  CHECK_ERR;
  cudaEventRecord(startT3);
  CHECK_ERR;
  sproduct_NT_NT_NT__d10_d21_22_10_d21_22__alpha_1_0_p_p_p__9329d0c(
      A_dev_begins_dev, 0, B_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els,
      nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopT3);
  CHECK_ERR;
  cudaEventSynchronize(stopT3);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT3, startT3, stopT3);
  CHECK_ERR;
  double elapsedTime = elapsedTimeT1 + elapsedTimeT2 + elapsedTimeT3;
  cudaDeviceSynchronize();
  CHECK_ERR;

  std::cout << "Gemmforge Tensor Contraction took: " << elapsedTime << " ms"
            << std::endl;
  cudaMemcpy(R1, A_dev, sizeof(float) * 4620 * num_els, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4620 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  double fp_per_el = 190344;
  double ls_per_el = 72036;
  double fp_unfused_per_el = 190344;
  double ls_unfused_per_el = 112692;
  fp_per_el *= num_els;
  ls_per_el *= num_els;
  fp_unfused_per_el *= num_els;
  ls_unfused_per_el *= num_els;
  std::cout << "Gemmforge Theoretical Fused Kernel GFLOPs/s: "
            << fp_per_el * 1e-6 / elapsedTime << std::endl;
  std::cout << "Operational Theoretical Fused intensity: "
            << fp_per_el / ls_per_el << std::endl;
  std::cout << "Gemmforge GFLOPs/s: " << fp_unfused_per_el * 1e-6 / elapsedTime
            << std::endl;
  std::cout << "Operational intensity: "
            << fp_unfused_per_el / ls_unfused_per_el << std::endl;
  double peakFLOPGiven = 29767.7;
  double peakBandwidthGiven = 760.08;

  if (peakFLOPGiven > 0.1 && peakBandwidthGiven) {
    double obtainable_peak =
        std::min(static_cast<double>(peakFLOPGiven),
                 static_cast<double>(peakBandwidthGiven *
                                     static_cast<double>(fp_per_el) /
                                     static_cast<double>(ls_per_el)));
    std::cout << 100.0 * (fp_per_el * 1e-6 / elapsedTime) / obtainable_peak
              << " % of roof w. respect to operational intensity achieved with "
                 "Gemmforge"
              << std::endl;
    // std::cout << 100.0*(fp_per_el * 1e-6 / elapsedTime) / obtainable_peak <<
    // " % of roof w. respect to operational intensity achieved with cuTensor"
    // << std::endl;
    double obtainable_unfused_peak =
        std::min(static_cast<double>(peakFLOPGiven),
                 static_cast<double>(peakBandwidthGiven *
                                     static_cast<double>(fp_unfused_per_el) /
                                     static_cast<double>(ls_unfused_per_el)));
    std::cout << 100.0 * (fp_unfused_per_el * 1e-6 / elapsedTime) /
                     obtainable_unfused_peak
              << " % of roof w. respect to unfused operational intensity "
                 "achieved with Gemmforge"
              << std::endl;
    // std::cout << 100.0*(fp_unfused_per_el * 1e-6 / elapsedTime) /
    // obtainable_unfused_peak << " % of roof w. respect to unfused operational
    // intensity achieved with cuTensor" << std::endl;
    double obtainable_unfused_peak_k1 = std::min(
        static_cast<double>(peakFLOPGiven),
        static_cast<double>(peakBandwidthGiven * static_cast<double>(56364) /
                            static_cast<double>(12340)));
    std::cout
        << 100.0 * (56364 * num_els * 1e-6 / elapsedTimeT1) /
               obtainable_unfused_peak_k1
        << " % of roof w. respect to Kernel1 intensity achieved with Gemmforge"
        << std::endl;
    double obtainable_unfused_peak_k2 = std::min(
        static_cast<double>(peakFLOPGiven),
        static_cast<double>(peakBandwidthGiven * static_cast<double>(124740) /
                            static_cast<double>(61504)));
    std::cout
        << 100.0 * (124740 * num_els * 1e-6 / elapsedTimeT2) /
               obtainable_unfused_peak_k2
        << " % of roof w. respect to Kernel2 intensity achieved with Gemmforge"
        << std::endl;
    double obtainable_unfused_peak_k3 = std::min(
        static_cast<double>(peakFLOPGiven),
        static_cast<double>(peakBandwidthGiven * static_cast<double>(9240) /
                            static_cast<double>(38848)));
    std::cout
        << 100.0 * (9240 * num_els * 1e-6 / elapsedTimeT3) /
               obtainable_unfused_peak_k3
        << " % of roof w. respect to Kernel3 intensity achieved with Gemmforge"
        << std::endl;
  }

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 4620 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 462 * num_els,
             cudaMemcpyHostToDevice);
  CHECK_ERR;

  if constexpr (!false) {
    cutensorHandle_t *handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    cudaEvent_t startCT1, stopCT1;
    cudaEvent_t startCT2, stopCT2;
    cudaEvent_t startCT3, stopCT3;
    cudaEventCreate(&startCT1);
    CHECK_ERR;
    cudaEventCreate(&stopCT1);
    CHECK_ERR;
    cudaEventCreate(&startCT2);
    CHECK_ERR;
    cudaEventCreate(&stopCT2);
    CHECK_ERR;
    cudaEventCreate(&startCT3);
    CHECK_ERR;
    cudaEventCreate(&stopCT3);
    CHECK_ERR;
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
      extent['k'] = 21;
      extent['l'] = 13;
      extent['m'] = 10;
      extent['p'] = 22;
      extent['q'] = 61;
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
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descA, nmodeA,
                                                extentA.data(), NULL, typeA,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descB;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descB, nmodeB,
                                                extentB.data(), NULL, typeB,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descC;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descC, nmodeC,
                                                extentC.data(), NULL, typeC,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descD;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descD, nmodeD,
                                                extentD.data(), NULL, typeD,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descE;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descE, nmodeE,
                                                extentE.data(), NULL, typeE,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descF;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descF, nmodeF,
                                                extentF.data(), NULL, typeF,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t descX;
      HANDLE_ERROR(cutensorInitTensorDescriptor(handle, &descX, nmodeX,
                                                extentX.data(), NULL, typeX,
                                                CUTENSOR_OP_IDENTITY));

      uint32_t alignmentRequirementA;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, A_dev, &descA,
                                                   &alignmentRequirementA));

      uint32_t alignmentRequirementB;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, B_dev, &descB,
                                                   &alignmentRequirementB));

      uint32_t alignmentRequirementC;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, C_dev, &descC,
                                                   &alignmentRequirementC));

      uint32_t alignmentRequirementD;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, D_dev, &descD,
                                                   &alignmentRequirementD));

      uint32_t alignmentRequirementE;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, E_dev, &descE,
                                                   &alignmentRequirementE));

      uint32_t alignmentRequirementF;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, F_dev, &descF,
                                                   &alignmentRequirementF));

      uint32_t alignmentRequirementX;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(handle, X_dev, &descX,
                                                   &alignmentRequirementX));

      cutensorContractionDescriptor_t desc1;
      HANDLE_ERROR(cutensorInitContractionDescriptor(
          handle, &desc1, &descC, modeC.data(), alignmentRequirementC, &descD,
          modeD.data(), alignmentRequirementD, &descX, modeX.data(),
          alignmentRequirementX, &descX, modeX.data(), alignmentRequirementX,
          typeCompute));

      cutensorContractionFind_t find1;
      HANDLE_ERROR(
          cutensorInitContractionFind(handle, &find1, CUTENSOR_ALGO_DEFAULT));

      uint64_t worksize1 = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspaceSize(
          handle, &desc1, &find1, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize1));

      cutensorContractionDescriptor_t desc2;
      HANDLE_ERROR(cutensorInitContractionDescriptor(
          handle, &desc2, &descF, modeF.data(), alignmentRequirementF, &descE,
          modeE.data(), alignmentRequirementE, &descA, modeA.data(),
          alignmentRequirementA, &descA, modeA.data(), alignmentRequirementA,
          typeCompute));

      cutensorContractionFind_t find2;
      HANDLE_ERROR(
          cutensorInitContractionFind(handle, &find2, CUTENSOR_ALGO_DEFAULT));

      uint64_t worksize2 = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspaceSize(
          handle, &desc2, &find2, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize2));

      cutensorContractionDescriptor_t desc3;
      HANDLE_ERROR(cutensorInitContractionDescriptor(
          handle, &desc3, &descB, modeB.data(), alignmentRequirementB, &descX,
          modeX.data(), alignmentRequirementX, &descA, modeA.data(),
          alignmentRequirementA, &descA, modeA.data(), alignmentRequirementA,
          typeCompute));

      cutensorContractionFind_t find3;
      HANDLE_ERROR(
          cutensorInitContractionFind(handle, &find3, CUTENSOR_ALGO_DEFAULT));

      uint64_t worksize3 = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspaceSize(
          handle, &desc3, &find3, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize3));

      uint64_t maxWorkSize =
          std::max(std::max(worksize1, worksize2), worksize3);
      void *work = nullptr;
      if (maxWorkSize > 0) {
        if (cudaSuccess != cudaMalloc(&work, maxWorkSize)) {
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
      HANDLE_ERROR(cutensorInitContractionPlan(handle, &plan1, &desc1, &find1,
                                               worksize1));

      cutensorContractionPlan_t plan2;
      HANDLE_ERROR(cutensorInitContractionPlan(handle, &plan2, &desc2, &find2,
                                               worksize2));

      cutensorContractionPlan_t plan3;
      HANDLE_ERROR(cutensorInitContractionPlan(handle, &plan3, &desc3, &find3,
                                               worksize3));

      cudaDeviceSynchronize();
      CHECK_ERR;

      cudaEventRecord(startCT1);
      CHECK_ERR;
      cutensorContraction(handle, &plan1, (void *)&alphaK1, C_dev, D_dev,
                          (void *)&betaK1, X_dev, X_dev, work, worksize1, 0);
      cudaEventRecord(stopCT1);
      CHECK_ERR;
      cudaEventSynchronize(stopCT1);
      CHECK_ERR;
      cudaEventElapsedTime(&elapsedTimeCT1, startCT1, stopCT1);
      CHECK_ERR;

      // cudaDeviceSynchronize(); CHECK_ERR;
      // cudaMemcpy(Ri1c, X_dev, sizeof(float) * 462 * num_els,
      // cudaMemcpyDeviceToHost); CHECK_ERR;

      cudaEventRecord(startCT2);
      CHECK_ERR;
      cutensorContraction(handle, &plan2, (void *)&alphaK2, F_dev, E_dev,
                          (void *)&betaK2, A_dev, A_dev, work, worksize2, 0);
      cudaEventRecord(stopCT2);
      CHECK_ERR;
      cudaEventSynchronize(stopCT2);
      CHECK_ERR;
      cudaEventElapsedTime(&elapsedTimeCT2, startCT2, stopCT2);
      CHECK_ERR;

      // cudaDeviceSynchronize(); CHECK_ERR;
      // cudaMemcpy(Ri2c, A_dev, sizeof(float) * 4620 * num_els,
      // cudaMemcpyDeviceToHost); CHECK_ERR;

      cudaEventRecord(startCT3);
      CHECK_ERR;
      cutensorContraction(handle, &plan3, (void *)&alphaK3, B_dev, X_dev,
                          (void *)&betaK3, A_dev, A_dev, work, worksize3, 0);
      cudaEventRecord(stopCT3);
      CHECK_ERR;
      cudaEventSynchronize(stopCT3);
      CHECK_ERR;
      cudaEventElapsedTime(&elapsedTimeCT3, startCT3, stopCT3);
      CHECK_ERR;

      cudaDeviceSynchronize();
      CHECK_ERR;

      cudaMemcpy(R2, A_dev, sizeof(float) * 4620 * num_els,
                 cudaMemcpyDeviceToHost);
      CHECK_ERR;

      cudaFree(work);
    }

    float elapsedTimeCuTensor =
        elapsedTimeCT1 + elapsedTimeCT2 + elapsedTimeCT2;
    if (peakFLOPGiven > 0.1 && peakBandwidthGiven) {
      double obtainable_peak =
          std::min(static_cast<double>(peakFLOPGiven),
                   static_cast<double>(peakBandwidthGiven *
                                       static_cast<double>(fp_per_el) /
                                       static_cast<double>(ls_per_el)));
      std::cout << 100.0 * (fp_per_el * 1e-6 / elapsedTimeCuTensor) /
                       obtainable_peak
                << " % of roof w. respect to operational intensity achieved "
                   "with cuTensor"
                << std::endl;

      double obtainable_unfused_peak =
          std::min(static_cast<double>(peakFLOPGiven),
                   static_cast<double>(peakBandwidthGiven *
                                       static_cast<double>(fp_unfused_per_el) /
                                       static_cast<double>(ls_unfused_per_el)));
      std::cout << 100.0 * (fp_unfused_per_el * 1e-6 / elapsedTimeCuTensor) /
                       obtainable_unfused_peak
                << " % of roof w. respect to unfused operational intensity "
                   "achieved with cuTensor"
                << std::endl;
    }

    /*
    bool i1results_wrong = false;
    for (size_t i = 0; i < 462 * num_els; i++){
      if (std::abs(Ri1[i] - Ri1c[i]) > 1.0f) {
        std::cout << "Intermediate Results 1 do not match, problem first at
    offset " << i << " :_(" << std::endl; i1results_wrong = true; break;
      }
    }
    if (!i1results_wrong){
      std::cout << "Gemmforge and cuTensor contraction intermediate results 1
    match! :)" << std::endl;
    }

    bool i2results_wrong = false;
    for (size_t i = 0; i < 4620 * num_els; i++){
      if (std::abs(Ri2[i] - Ri2c[i]) > 1.0f) {
        std::cout << "Intermediate Results 2 do not match, problem first at
    offset " << i << " :_(" << std::endl; i2results_wrong = true; break;
      }
    }
    if (!i2results_wrong){
      std::cout << "Gemmforge and cuTensor contraction intermediate results 2
    match! :)" << std::endl;
    }
    */

    bool results_wrong = false;
    for (size_t i = 0; i < 4620 * num_els; i++) {
      if (std::abs(R1[i] - R2[i]) > 5.0f) {
        std::cout << "Results do not match, problem first at offset " << i
                  << " :_(" << std::endl;
        results_wrong = true;
        break;
      }
    }
    if (!results_wrong) {
      std::cout << "Gemmforge and cuTensor contraction results match! :)"
                << std::endl;
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
