
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
 kernel_sloopOverGEMM_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(9, 46), bbox=BoundingBox(Range(0, 9), Range(0, 46)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 46), size=414, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(9, 26), bbox=BoundingBox(Range(0, 9), Range(0, 26)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 26), size=234, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(26, 46), bbox=BoundingBox(Range(0, 26), Range(0, 46)), stride=(1, 26), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(26, 46), size=1196, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 9), Range(0, 46), Range(0, 26))), 'matrix_a': DenseMatrix{name = C, num. rows = 9, num. columns = 26, leading dimension = 9, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 9, 26]}, 'matrix_b': DenseMatrix{name = D, num. rows = 26, num. columns = 46, leading dimension = 26, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 26, 46]}, 'matrix_c': DenseMatrix{name = X, num. rows = 9, num. columns = 46, leading dimension = 9, direction = DataFlowDirection.SINK, bbox = [0, 0, 9, 46]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(9, 46), bbox=BoundingBox(Range(0, 9), Range(0, 46)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 46), size=414, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(9, 26), bbox=BoundingBox(Range(0, 9), Range(0, 26)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 26), size=234, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(26, 46), bbox=BoundingBox(Range(0, 26), Range(0, 46)), stride=(1, 26), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(26, 46), size=1196, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 9), Range(0, 46), Range(0, 26))), 'matrix_a': DenseMatrix{name = C, num. rows = 9, num. columns = 26, leading dimension = 9, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 9, 26]}, 'matrix_b': DenseMatrix{name = D, num. rows = 26, num. columns = 46, leading dimension = 26, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 26, 46]}, 'matrix_c': DenseMatrix{name = X, num. rows = 9, num. columns = 46, leading dimension = 9, direction = DataFlowDirection.SINK, bbox = [0, 0, 9, 46]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
        const float * const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        const float * const __restrict__ glb_D = &D[batchID][0 + D_extraOffset];
        float reg0[46] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __shared__  __align__(8) float totalShrMem[1430];
        float * localShrMem0 = &totalShrMem[1430 * threadIdx.y];
        
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 7; ++i) {
            shrRegion0[threadIdx.x + i * 32] = glb_C[threadIdx.x + i * 32];
          }
          if (threadIdx.x < 10) {
            shrRegion0[threadIdx.x + 224] = glb_C[threadIdx.x + 224];
          }
        }
        
        float* shrRegion1 = &localShrMem0[234];
        // using ExtendedPatchLoader
        {
          #pragma unroll
          for (int i = 0; i < 37; ++i) {
            shrRegion1[threadIdx.x + i * 32] = glb_D[threadIdx.x + i * 32];
          }
          if (threadIdx.x < 12) {
            shrRegion1[threadIdx.x + 1184] = glb_D[threadIdx.x + 1184];
          }
        }
        __syncwarp();
        if (threadIdx.x < 9) {
          float value;
        
          #pragma unroll
          for (int k = 0; k < 26; ++k) {
            value = shrRegion0[threadIdx.x + k * 9];
        
            #pragma unroll
            for (int n = 0; n < 46; ++n) {
              reg0[n] += value * shrRegion1[k + 26 * n];
            }
          }
        }
        if (threadIdx.x < 9) {
          #pragma unroll
          for (int n = 0; n < 46; ++n) {
            glb_X[threadIdx.x + 9 * n] = reg0[n];
          }
        }
        
      }
    }
  }
}
void sloopOverGEMM_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3<<<grid,block,0,stream>>>(C, C_extraOffset, D, D_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}


__global__ void 
__launch_bounds__(64)
 kernel_sloopOverGEMM2_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      /*
      This is the LoG created from the following YaTeTo description:
      ('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(9, 46), bbox=BoundingBox(Range(0, 9), Range(0, 46)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 46), size=414, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(9, 26), bbox=BoundingBox(Range(0, 9), Range(0, 26)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 26), size=234, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(26, 46), bbox=BoundingBox(Range(0, 26), Range(0, 46)), stride=(1, 26), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(26, 46), size=1196, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 9), Range(0, 46), Range(0, 26))), 'matrix_a': DenseMatrix{name = C, num. rows = 9, num. columns = 26, leading dimension = 9, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 9, 26]}, 'matrix_b': DenseMatrix{name = D, num. rows = 26, num. columns = 46, leading dimension = 26, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 26, 46]}, 'matrix_c': DenseMatrix{name = X, num. rows = 9, num. columns = 46, leading dimension = 9, direction = DataFlowDirection.SINK, bbox = [0, 0, 9, 46]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
      */
      {
    //('gemm', {'descr': Description(  result=TensorDescription(  name=X,	  memoryLayout=DenseMemoryLayout(shape=(9, 46), bbox=BoundingBox(Range(0, 9), Range(0, 46)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 46), size=414, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  leftTerm=TensorDescription(  name=C,	  memoryLayout=DenseMemoryLayout(shape=(9, 26), bbox=BoundingBox(Range(0, 9), Range(0, 26)), stride=(1, 9), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(9, 26), size=234, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  rightTerm=TensorDescription(  name=D,	  memoryLayout=DenseMemoryLayout(shape=(26, 46), bbox=BoundingBox(Range(0, 26), Range(0, 46)), stride=(1, 26), align=<yateto.arch.Architecture object at 0x7fa6552a76d0>),	  eqspp=dense(shape=(26, 46), size=1196, ndim=2),	  is_compute_constant=False,	  is_temporary=False),	  transA=False,	  transB=False,	  alpha=1.0,	  beta=0.0,	  prefetchName=None,	  isACsc=False,	  isBCsc=False,	  alignedA=False,	  alignedC=False,	  mnk=(Range(0, 9), Range(0, 46), Range(0, 26))), 'matrix_a': DenseMatrix{name = C, num. rows = 9, num. columns = 26, leading dimension = 9, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 9, 26]}, 'matrix_b': DenseMatrix{name = D, num. rows = 26, num. columns = 46, leading dimension = 26, direction = DataFlowDirection.SOURCE, bbox = [0, 0, 26, 46]}, 'matrix_c': DenseMatrix{name = X, num. rows = 9, num. columns = 46, leading dimension = 9, direction = DataFlowDirection.SINK, bbox = [0, 0, 9, 46]}, 'args': ['C, extraOffset_C', 'D, extraOffset_D', 'X, extraOffset_X', 'numElements', 'flags', 'streamPtr']})
        const float * const __restrict__ glb_C = &C[batchID][0 + C_extraOffset];
        float * const __restrict__ glb_X = &X[batchID][0 + X_extraOffset];
        const float * const __restrict__ glb_D = &D[batchID][0 + D_extraOffset];
        float reg0[9] = {0.f};
        __shared__  __align__(8) float totalShrMem[1430];
        float * localShrMem0 = &totalShrMem[1430 * threadIdx.y];
        __shared__ __align__(8) float shr_X[46*9];
        float* shrRegion0 = &localShrMem0[0];
        // using ExtendedPatchLoader
        {
          #pragma unroll 3
          for (int i = 0; i < 3; ++i) {
            shrRegion0[threadIdx.x + i * 64] = glb_C[threadIdx.x + i * 64];
          }
          if (threadIdx.x < 42) {
            shrRegion0[threadIdx.x + 64*3] = glb_C[threadIdx.x + 64*3];
          }
        }
        
        float* shrRegion1 = &localShrMem0[234];
        // using ExtendedPatchLoader
        {
          #pragma unroll 18
          for (int i = 0; i < 18; ++i) {
            shrRegion1[threadIdx.x + i * 64] = glb_D[threadIdx.x + i * 64];
          }
          if (threadIdx.x < 44) {
            shrRegion1[threadIdx.x + 64*18] = glb_D[threadIdx.x + 64*18];
          }
        }

        // M = 9, K = 26, N = 46
        __syncwarp();
        if (threadIdx.x < 46) {
          float value;

          #pragma unroll 26
          for (int k = 0; k < 26; ++k) {
            value = shrRegion1[threadIdx.x * 26 + k];
        
            #pragma unroll 9
            for (int m = 0; m < 9; ++m) {
              reg0[m] += value * shrRegion0[k*9 + m];
            }
          }
        }

        if (threadIdx.x < 46) {
          #pragma unroll 9
          for (int m = 0; m < 9; ++m) {
            shr_X[threadIdx.x*9 + m] = reg0[m];
          }
        }
        __syncwarp();

        #pragma unroll
        for (int i =  0; i < 6; i++){
          glb_X[threadIdx.x + i*64] = shr_X[threadIdx.x + i*64];
        }
        if (threadIdx.x < 30){
          glb_X[threadIdx.x + 6*64] = shr_X[threadIdx.x + 6*64];
        }
      }
    }
  }
}
void sloopOverGEMM2_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(const float * const * C, int C_extraOffset, const float * const * D, int D_extraOffset, float ** X, int X_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(64, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sloopOverGEMM2_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3<<<grid,block,0,stream>>>(C, C_extraOffset, D, D_extraOffset, X, X_extraOffset, numElements, flags);
  CHECK_ERR;
}

int main(){
  constexpr size_t num_els = 19072;
  float* A = new float[7038 * num_els]{0.f};
  float* B = new float[17 * num_els]{0.f};
  float* C = new float[234 * num_els]{0.f};
  float* D = new float[1196 * num_els]{0.f};
  float* E = new float[36846 * num_els]{0.f};
  float* F = new float[1513 * num_els]{0.f};
  float* X = new float[414 * num_els]{0.f};
  float* R1 = new float[414 * num_els]{0.f};
  float* R2 = new float[414 * num_els]{0.f};

  float* coreA = new float[7038];
  float* coreB = new float[17];
  float* coreC = new float[234];
  float* coreD = new float[1196];
  float* coreE = new float[36846];
  float* coreF = new float[1513];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, 100);
  for (size_t i = 0; i < 7038; i++){
    coreA[i] = distribution(gen);
  }
  for (size_t i = 0; i < 17; i++){
    coreB[i] = distribution(gen);
  }
  for (size_t i = 0; i < 234; i++){
    coreC[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1196; i++){
    coreD[i] = distribution(gen);
  }
  for (size_t i = 0; i < 36846; i++){
    coreE[i] = distribution(gen);
  }
  for (size_t i = 0; i < 1513; i++){
    coreF[i] = distribution(gen);
  }

  for (size_t i = 0; i < num_els; i++){
      std::memcpy(&A[i * 7038], &coreA[0], 7038 * sizeof(float));
      std::memcpy(&B[i * 17], &coreB[0], 17 * sizeof(float));
      std::memcpy(&C[i * 234], &coreC[0], 234 * sizeof(float));
      std::memcpy(&D[i * 1196], &coreD[0], 1196 * sizeof(float));
      std::memcpy(&E[i * 36846], &coreE[0], 36846 * sizeof(float));
      std::memcpy(&F[i * 1513], &coreF[0], 1513 * sizeof(float));
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

  cudaMalloc((void **)&A_dev, sizeof(float) * 7038 * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev, sizeof(float) * 17 * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev, sizeof(float) * 234 * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev, sizeof(float) * 1196 * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev, sizeof(float) * 36846 * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev, sizeof(float) * 1513 * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev, sizeof(float) * 414 * num_els); CHECK_ERR;

  cudaMalloc((void **)&A_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&B_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&C_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&D_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&E_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&F_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
  cudaMalloc((void **)&X_dev_begins_dev, sizeof(float*) * num_els); CHECK_ERR;
 
  cudaDeviceSynchronize(); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 7038 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 17 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev, (void *)C, sizeof(float) * 234 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev, (void *)D, sizeof(float) * 1196 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev, (void *)E, sizeof(float) * 36846 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev, (void *)F, sizeof(float) * 1513 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 414 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  for (size_t i = 0; i < num_els; i++){
    A_dev_begins[i] = A_dev + i * 7038;
    B_dev_begins[i] = B_dev + i * 17;
    C_dev_begins[i] = C_dev + i * 234;
    D_dev_begins[i] = D_dev + i * 1196;
    E_dev_begins[i] = E_dev + i * 36846;
    F_dev_begins[i] = F_dev + i * 1513;
    X_dev_begins[i] = X_dev + i * 414;
  }

  cudaMemcpy((void *)A_dev_begins_dev, (void *)A_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dev_begins_dev, (void *)B_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C_dev_begins_dev, (void *)C_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)D_dev_begins_dev, (void *)D_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)E_dev_begins_dev, (void *)E_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)F_dev_begins_dev, (void *)F_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)X_dev_begins_dev, (void *)X_dev_begins, sizeof(float*) * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  sloopOverGEMM_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaDeviceSynchronize(); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 414 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;

  std::cout << "Dimensions: " << 9 << ", " << 89 << ", " << 17 << ", " << 46 << ", " << 26 << ", " << 89 << std::endl;

  float elapsedTimeT1 = 0.0;
  cudaEvent_t startT1, stopT1;
  cudaEventCreate(&startT1); CHECK_ERR;
  cudaEventCreate(&stopT1); CHECK_ERR;
  cudaEventRecord(startT1); CHECK_ERR;
  sloopOverGEMM_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT1); CHECK_ERR;
  cudaEventSynchronize(stopT1); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT1, startT1, stopT1); CHECK_ERR;


  std::cout << "Gemmforge Tensor 1 Contraction took: " << elapsedTimeT1 << " ms" << std::endl; 
  cudaMemcpy(R1, X_dev, sizeof(float) * 414 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 414 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  float elapsedTimeT2 = 0.0;
  cudaEvent_t startT2, stopT2;
  cudaEventCreate(&startT2); CHECK_ERR;
  cudaEventCreate(&stopT2); CHECK_ERR;
  cudaEventRecord(startT2); CHECK_ERR;
  sloopOverGEMM2_NT_NT_NT__d9_26_d9_46_d26_46__alpha_1_0_beta_0_0_p_p_p__fe4a7c3(C_dev_begins_dev, 0, D_dev_begins_dev, 0, X_dev_begins_dev, 0, num_els, nullptr, nullptr); CHECK_ERR;
  cudaEventRecord(stopT2); CHECK_ERR;
  cudaEventSynchronize(stopT2); CHECK_ERR;
  cudaEventElapsedTime(&elapsedTimeT2, startT2, stopT2); CHECK_ERR;


  std::cout << "Gemmforge Tensor 2 Contraction took: " << elapsedTimeT2 << " ms" << std::endl; 
  cudaMemcpy(R2, X_dev, sizeof(float) * 414 * num_els, cudaMemcpyDeviceToHost); CHECK_ERR;
  cudaMemcpy((void *)X_dev, (void *)X, sizeof(float) * 414 * num_els, cudaMemcpyHostToDevice); CHECK_ERR;


  bool results_wrong = false;
  for (size_t i = 0; i < 414 * num_els; i++){
    if (std::abs(R1[i] - R2[i]) > 10.0f) {
      std::cout << "Results do not match, problem first at offset " << i << " :_(" << std::endl;
      results_wrong = true;
      break;
    }
  }
  if (!results_wrong){
    std::cout << "Gemmforge and cuTensor contraction results match! :)" << std::endl;
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

