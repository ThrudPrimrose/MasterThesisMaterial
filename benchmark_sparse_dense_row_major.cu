#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <random>
#include <iomanip>

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
        const int line)
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
        throw;
    }
    PrevFile = File;
    PrevLine = Line;
#endif
}

__global__ void 
__launch_bounds__(32)
 kernel_col_major(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f};
      __shared__  __align__(8) float totalShrMem[1024];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x * 32 + i];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32) {
        float value;

        for (int k = 0; k < 32; k++) {
          value = glb_B[threadIdx.x * 32 + k];

          #pragma unroll
          for (int m = 0; m < 32; ++m) {
            reg0[m] += value * shrRegion0[m + 32 * k];
          }
        }
      }
      if (threadIdx.x < 32) {
        #pragma unroll
        for (int m = 0; m < 32; ++m) {
          glb_C[threadIdx.x * 32 + m] = reg0[m] + glb_C[threadIdx.x * 32 + m];
        }
      }
    }
  }
}

__global__ void 
__launch_bounds__(32)
 kernel_row_major(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float * const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float * const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f};
      __shared__  __align__(8) float totalShrMem[1024];
      float * localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x * 32 + i];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32) {
        float value;

        for (int k = 0; k < 32; k++) {
          value = glb_B[threadIdx.x * 32 + k];

          #pragma unroll
          for (int m = 0; m < 32; ++m) {
            reg0[m] += value * shrRegion0[m + 32 * k];
          }
        }
      }
      if (threadIdx.x < 32) {
        #pragma unroll
        for (int m = 0; m < 32; ++m) {
          glb_C[threadIdx.x * 32 + m] = reg0[m] + glb_C[threadIdx.x * 32 + m];
        }
      }
    }
  }
}

void call_row_major(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_row_major<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

void call_col_major(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_col_major<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

// Function to print the matrix in column-major format
void printMatrixCM(const std::vector<double>& matrix, int rows, int cols) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            std::cout << matrix[i + j * rows] << "\t";
        }
        std::cout << std::endl;
    }
}

// Function to print the matrix in row-major format
void printMatrixRM(const std::vector<double>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void printMatrixCM(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i + j * rows] << "\t";
        }
        std::cout << std::endl;
    }
}

// Function to print the matrix in row-major format
void printMatrixRM(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

// Function to transpose a column-major matrix to row-major format
void transposeCMtoRM(const float* source, float* destination, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            destination[i * cols + j] = source[i + j * rows];
        }
    }
}

int main() {
    std::cout <<  std::fixed << std::setw(5) << std::setprecision(3);
    const int rows = 32;
    const int cols = 32;
    const int numElements = 1 * rows * cols;

    // Create column-major matrices A_cm, B_cm, C_cm
    float A_cm[numElements];
    float B_cm[numElements];
    float C_cm[numElements];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Fill column-major matrices with random values
    for (int i = 0; i < numElements; ++i) {
        A_cm[i] = dis(gen);
        B_cm[i] = dis(gen);
        C_cm[i] = dis(gen);
    }

    // Transpose column-major matrices to row-major format
    float A_rm[numElements];
    float B_rm[numElements];
    float C_rm[numElements];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A_rm[i * cols + j] = A_cm[i + j * rows];
            B_rm[i * cols + j] = B_cm[i + j * rows];
            C_rm[i * cols + j] = C_cm[i + j * rows];
        }
    }

    float *d_A_cm = nullptr;
    float *d_B_cm = nullptr;
    float *d_C_cm = nullptr;
    float *d_A_rm = nullptr;
    float *d_B_rm = nullptr;
    float *d_C_rm = nullptr;

    std::vector<float*> devArrays{d_A_cm, d_B_cm, d_C_cm, d_A_rm, d_B_rm, d_C_rm};
    std::vector<float*> hostArrays{A_cm, B_cm, C_cm, A_rm, B_rm, C_rm};

    for (int i = 0; i < devArrays.size(); i++){
        float* devArray = devArrays[i];
        float* hostArray = hostArrays[i];
        cudaMalloc((void**)&devArray, numElements * sizeof(float));
        cudaMemcpy(devArray, hostArray, numElements * sizeof(float), cudaMemcpyHostToDevice);
    }

    call_col_major(A_cm, 0, B_cm, 0, C_cm, 0, 1, nullptr, nullptr);
    call_row_major(A_rm, 0, B_rm, 0, C_rm, 0, 1, nullptr, nullptr);

    cudaMemcpy(C_rm, d_C_rm, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cm, d_C_cm, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Row-Major Matrix C_rm:" << std::endl;
    printMatrixRM(C_rm, rows, cols);
    std::cout << std::endl;

    std::cout << "Col-Major Matrix C_cm:" << std::endl;
    printMatrixCM(C_cm, rows, cols);
    std::cout << std::endl;

    return 0;
}