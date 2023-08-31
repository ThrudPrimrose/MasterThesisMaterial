#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include <cassert>
#include <cooperative_groups.h>

#define CHECK_ERR checkErr(__FILE__, __LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
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

void checkErr(const std::string &File, int Line)
{
#ifndef NDEBUG
  cudaError_t Error = cudaGetLastError();
  if (Error != cudaSuccess)
  {
    std::cout << std::endl
              << File
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
        kernel_A_B_DenseXDense(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__ __align__(8) float totalShrMem[1024*2];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x + i * 32] = glb_B[threadIdx.x + i * 32];
        }
      }

      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; ++k)
        {
          value = glb_A[threadIdx.x + k * 32];

#pragma unroll
          for (int n = 0; n < 32; ++n)
          {
            reg0[n] += value * shrRegion0[k + 32 * n];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          glb_C[threadIdx.x + 32 * n] = reg0[n] + glb_C[threadIdx.x + 32 * n];
        }
      }
    }
  }
}

__global__ void
    __launch_bounds__(32)
        kernel_A_B_DenseXDense_CO(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__ __align__(8) float totalShrMem[1024*3];
      float *localShrMem0 = &totalShrMem[1024 * 3 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      float *shrRegion1 = &localShrMem0[1024];
      float *shrRegion2 = &localShrMem0[2048];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x + i * 32];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion1[threadIdx.x * 32 + i] = glb_B[threadIdx.x + i * 32];
        }
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion2[threadIdx.x * 32 + i] = glb_C[threadIdx.x + i * 32];
        }
      }

      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; ++k)
        {
          value = shrRegion1[threadIdx.x + k * 32];

#pragma unroll
          for (int n = 0; n < 32; ++n)
          {
            reg0[n] += value * shrRegion0[k + 32 * n];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          shrRegion2[threadIdx.x + 32 * n] += reg0[n];
        }
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          glb_C[threadIdx.x + 32 * n] = shrRegion2[threadIdx.x * 32 + n];
        }
      }
    }
  }
}


__global__ void
    __launch_bounds__(32)
        kernel_A_B_DenseXDense_row_major_dist_A(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__ __align__(8) float totalShrMem[1024];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x + i * 32] = glb_B[threadIdx.x + i * 32];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; ++k)
        {
          value = glb_A[threadIdx.x * 32 + k];

#pragma unroll
          for (int n = 0; n < 32; ++n)
          {
            reg0[n] += value * shrRegion0[k * 32 + n];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          glb_C[threadIdx.x * 32 + n] = reg0[n] + glb_C[threadIdx.x * 32 + n];
        }
      }
    }
  }
}

__global__ void
    __launch_bounds__(32)
        kernel_A_B_DenseXDense_row_major_dist_B(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__ __align__(8) float totalShrMem[1024];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x + i * 32] = glb_B[threadIdx.x + i * 32];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; ++k)
        {
          value = glb_A[threadIdx.x * 32 + k];

#pragma unroll
          for (int n = 0; n < 32; ++n)
          {
            reg0[n] += value * shrRegion0[k * 32 + n];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          glb_C[threadIdx.x * 32 + n] = reg0[n] + glb_C[threadIdx.x * 32 + n];
        }
      }
    }
  }
}

__constant__ int r[32][64] = {
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
};

__constant__ int offsets[32][64] = {
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
{32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
{64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95},
{96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127},
{128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159},
{160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191},
{192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223},
{224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255},
{256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287},
{288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319},
{320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351},
{352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383},
{384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415},
{416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447},
{448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479},
{480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511},
{512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543},
{544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575},
{576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607},
{608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639},
{640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671},
{672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703},
{704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735},
{736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767},
{768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799},
{800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831},
{832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863},
{864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895},
{896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927},
{928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959},
{960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991},
{992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023},
};

// Dense x Sparse Kernel
__global__ void
    __launch_bounds__(32)
        kernel_A_full_B_SparseXDense_Ellpack(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f};
      __shared__ __align__(8) float totalShrMem[1024];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x + i * 32] = glb_B[threadIdx.x + i * 32];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        #pragma unroll
        for (int k : r[threadIdx.x])
        {
          value = glb_A[threadIdx.x + k * 32];

          #pragma unroll
          for (int n = 0; n < 32; n++){
            reg0[n] += value * shrRegion0[k * 32 + threadIdx.x];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
          glb_C[threadIdx.x + 32 * n] = reg0[n] + glb_C[threadIdx.x + 32 * n];
        }
      }
    }
  }
}

// Dense x Sparse Kernel
__global__ void
    __launch_bounds__(32)
        kernel_A_full_B_SparseXDense_Distribute_Columns_of_B(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f};
      __shared__ __align__(8) float totalShrMem[1024];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x * 32 + i];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; k++)
        {
          // int col = r[threadIdx.x][k*2 + 1];
          value = glb_B[threadIdx.x * 32 + k];

#pragma unroll
          for (int m = 0; m < 32; ++m)
          {
            reg0[m] += value * shrRegion0[m + 32 * k];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int m = 0; m < 32; ++m)
        {
          glb_C[threadIdx.x * 32 + m] = reg0[m] + glb_C[threadIdx.x * 32 + m];
        }
      }
    }
  }
}

// Dense x Sparse Kernel
__global__ void
    __launch_bounds__(32)
        kernel_A_full_B_SparseXDense_Distribute_Rows_of_B(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32 * 32] = {0.0f};
      __shared__ __align__(8) float totalShrMem[1024];
      float *localShrMem0 = &totalShrMem[1024 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x * 32 + i];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int n = 0; n < 32; n++)
        {
          value = glb_B[threadIdx.x * 32 + n];

#pragma unroll
          for (int m = 0; m < 32; ++m)
          {
            reg0[m + 32 * n] += value * shrRegion0[m + 32 * n];
          }
        }
      }
      if (threadIdx.x < 32)
      {
// Atomic operation required, or an efficient implementation like binary reduce,
// Even this way it is slow as fuck
#pragma unroll
        for (int m = 0; m < 32 * 32; ++m)
        {
          // glb_C[m] = reg0[m] + glb_C[m];
          atomicAdd(&glb_C[m], reg0[m]);
        }
      }
    }
  }
}


// Dense x Sparse Kernel
__global__ void
    __launch_bounds__(32)
        kernel_A_full_B_SparseXDense_Distribute_Cols_of_B_double_load(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags)
{
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements)
  {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed)
    {
      const float *const __restrict__ glb_A = &A[batchID * 1024 + 0 + A_extraOffset];
      const float *const __restrict__ glb_B = &B[batchID * 1024 + 0 + B_extraOffset];
      float *const __restrict__ glb_C = &C[batchID * 1024 + 0 + C_extraOffset];
      float reg0[32] = {0.0f};
      __shared__ __align__(8) float totalShrMem[1024 * 2];
      float *localShrMem0 = &totalShrMem[1024 * 2 * threadIdx.y];

      float *shrRegion0 = &localShrMem0[0];
      float *shrRegion1 = &localShrMem0[1024];
      // using ExtendedPatchLoader
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion0[threadIdx.x * 32 + i] = glb_A[threadIdx.x * 32 + i];
        }
      }
      {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          shrRegion1[threadIdx.x * 32 + i] = glb_C[threadIdx.x * 32 + i];
        }
      }
      __syncwarp();
      if (threadIdx.x < 32)
      {
        float value;

        for (int k = 0; k < 32; k++)
        {
          value = glb_B[threadIdx.x * 32 + k];

#pragma unroll
          for (int m = 0; m < 32; ++m)
          {
            reg0[m] += value * shrRegion0[m + 32 * k];
          }
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int m = 0; m < 32; ++m)
        {
          shrRegion1[threadIdx.x * 32 + m] = reg0[m];
        }
      }
      if (threadIdx.x < 32)
      {
#pragma unroll
        for (int m = 0; m < 32; ++m)
        {
          glb_C[threadIdx.x + 32 * m] = shrRegion1[threadIdx.x + 32 * m];
        }
      }
    }
  }
}

// Dense x Dense Kernel Launcher
void A_B_DenseXDense(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_B_DenseXDense<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

// Dense x Dense Kernel Launcher
void A_B_DenseXDense_CO(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_B_DenseXDense_CO<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}


// Dense x Sparse Kernel Launcher
void A_full_B_SparseXDense_Ellpack(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_full_B_SparseXDense_Ellpack<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

void A_full_B_SparseXDense_Distribute_Columns_of_B(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_full_B_SparseXDense_Distribute_Columns_of_B<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

void A_full_B_SparseXDense_Distribute_Rows_of_B(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_full_B_SparseXDense_Distribute_Rows_of_B<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

void A_full_B_SparseXDense_Distribute_Cols_of_B_double_load(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned *flags, void *streamPtr)
{
  dim3 block(32, 1, 1);
  dim3 grid((numElements + 1 - 1) / 1, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_A_full_B_SparseXDense_Distribute_Cols_of_B_double_load<<<grid, block, 0, stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}

int main()
{
  std::cout << "Gemm-Type: "
            << "A_full_B_SparseXDense" << std::endl;
  // Element Matrices
  std::cout << "Instantiating core matrices" << std::endl;
  float CoreA[32 * 32] = {1.000e+00, 2.000e+00, 3.000e+00, 4.000e+00, 5.000e+00, 6.000e+00,
                          7.000e+00, 8.000e+00, 9.000e+00, 1.000e+01, 1.100e+01, 1.200e+01,
                          1.300e+01, 1.400e+01, 1.500e+01, 1.600e+01, 1.700e+01, 1.800e+01,
                          1.900e+01, 2.000e+01, 2.100e+01, 2.200e+01, 2.300e+01, 2.400e+01,
                          2.500e+01, 2.600e+01, 2.700e+01, 2.800e+01, 2.900e+01, 3.000e+01,
                          3.100e+01, 3.200e+01, 3.300e+01, 3.400e+01, 3.500e+01, 3.600e+01,
                          3.700e+01, 3.800e+01, 3.900e+01, 4.000e+01, 4.100e+01, 4.200e+01,
                          4.300e+01, 4.400e+01, 4.500e+01, 4.600e+01, 4.700e+01, 4.800e+01,
                          4.900e+01, 5.000e+01, 5.100e+01, 5.200e+01, 5.300e+01, 5.400e+01,
                          5.500e+01, 5.600e+01, 5.700e+01, 5.800e+01, 5.900e+01, 6.000e+01,
                          6.100e+01, 6.200e+01, 6.300e+01, 6.400e+01, 6.500e+01, 6.600e+01,
                          6.700e+01, 6.800e+01, 6.900e+01, 7.000e+01, 7.100e+01, 7.200e+01,
                          7.300e+01, 7.400e+01, 7.500e+01, 7.600e+01, 7.700e+01, 7.800e+01,
                          7.900e+01, 8.000e+01, 8.100e+01, 8.200e+01, 8.300e+01, 8.400e+01,
                          8.500e+01, 8.600e+01, 8.700e+01, 8.800e+01, 8.900e+01, 9.000e+01,
                          9.100e+01, 9.200e+01, 9.300e+01, 9.400e+01, 9.500e+01, 9.600e+01,
                          9.700e+01, 9.800e+01, 9.900e+01, 1.000e+02, 1.010e+02, 1.020e+02,
                          1.030e+02, 1.040e+02, 1.050e+02, 1.060e+02, 1.070e+02, 1.080e+02,
                          1.090e+02, 1.100e+02, 1.110e+02, 1.120e+02, 1.130e+02, 1.140e+02,
                          1.150e+02, 1.160e+02, 1.170e+02, 1.180e+02, 1.190e+02, 1.200e+02,
                          1.210e+02, 1.220e+02, 1.230e+02, 1.240e+02, 1.250e+02, 1.260e+02,
                          1.270e+02, 1.280e+02, 1.290e+02, 1.300e+02, 1.310e+02, 1.320e+02,
                          1.330e+02, 1.340e+02, 1.350e+02, 1.360e+02, 1.370e+02, 1.380e+02,
                          1.390e+02, 1.400e+02, 1.410e+02, 1.420e+02, 1.430e+02, 1.440e+02,
                          1.450e+02, 1.460e+02, 1.470e+02, 1.480e+02, 1.490e+02, 1.500e+02,
                          1.510e+02, 1.520e+02, 1.530e+02, 1.540e+02, 1.550e+02, 1.560e+02,
                          1.570e+02, 1.580e+02, 1.590e+02, 1.600e+02, 1.610e+02, 1.620e+02,
                          1.630e+02, 1.640e+02, 1.650e+02, 1.660e+02, 1.670e+02, 1.680e+02,
                          1.690e+02, 1.700e+02, 1.710e+02, 1.720e+02, 1.730e+02, 1.740e+02,
                          1.750e+02, 1.760e+02, 1.770e+02, 1.780e+02, 1.790e+02, 1.800e+02,
                          1.810e+02, 1.820e+02, 1.830e+02, 1.840e+02, 1.850e+02, 1.860e+02,
                          1.870e+02, 1.880e+02, 1.890e+02, 1.900e+02, 1.910e+02, 1.920e+02,
                          1.930e+02, 1.940e+02, 1.950e+02, 1.960e+02, 1.970e+02, 1.980e+02,
                          1.990e+02, 2.000e+02, 2.010e+02, 2.020e+02, 2.030e+02, 2.040e+02,
                          2.050e+02, 2.060e+02, 2.070e+02, 2.080e+02, 2.090e+02, 2.100e+02,
                          2.110e+02, 2.120e+02, 2.130e+02, 2.140e+02, 2.150e+02, 2.160e+02,
                          2.170e+02, 2.180e+02, 2.190e+02, 2.200e+02, 2.210e+02, 2.220e+02,
                          2.230e+02, 2.240e+02, 2.250e+02, 2.260e+02, 2.270e+02, 2.280e+02,
                          2.290e+02, 2.300e+02, 2.310e+02, 2.320e+02, 2.330e+02, 2.340e+02,
                          2.350e+02, 2.360e+02, 2.370e+02, 2.380e+02, 2.390e+02, 2.400e+02,
                          2.410e+02, 2.420e+02, 2.430e+02, 2.440e+02, 2.450e+02, 2.460e+02,
                          2.470e+02, 2.480e+02, 2.490e+02, 2.500e+02, 2.510e+02, 2.520e+02,
                          2.530e+02, 2.540e+02, 2.550e+02, 2.560e+02, 2.570e+02, 2.580e+02,
                          2.590e+02, 2.600e+02, 2.610e+02, 2.620e+02, 2.630e+02, 2.640e+02,
                          2.650e+02, 2.660e+02, 2.670e+02, 2.680e+02, 2.690e+02, 2.700e+02,
                          2.710e+02, 2.720e+02, 2.730e+02, 2.740e+02, 2.750e+02, 2.760e+02,
                          2.770e+02, 2.780e+02, 2.790e+02, 2.800e+02, 2.810e+02, 2.820e+02,
                          2.830e+02, 2.840e+02, 2.850e+02, 2.860e+02, 2.870e+02, 2.880e+02,
                          2.890e+02, 2.900e+02, 2.910e+02, 2.920e+02, 2.930e+02, 2.940e+02,
                          2.950e+02, 2.960e+02, 2.970e+02, 2.980e+02, 2.990e+02, 3.000e+02,
                          3.010e+02, 3.020e+02, 3.030e+02, 3.040e+02, 3.050e+02, 3.060e+02,
                          3.070e+02, 3.080e+02, 3.090e+02, 3.100e+02, 3.110e+02, 3.120e+02,
                          3.130e+02, 3.140e+02, 3.150e+02, 3.160e+02, 3.170e+02, 3.180e+02,
                          3.190e+02, 3.200e+02, 3.210e+02, 3.220e+02, 3.230e+02, 3.240e+02,
                          3.250e+02, 3.260e+02, 3.270e+02, 3.280e+02, 3.290e+02, 3.300e+02,
                          3.310e+02, 3.320e+02, 3.330e+02, 3.340e+02, 3.350e+02, 3.360e+02,
                          3.370e+02, 3.380e+02, 3.390e+02, 3.400e+02, 3.410e+02, 3.420e+02,
                          3.430e+02, 3.440e+02, 3.450e+02, 3.460e+02, 3.470e+02, 3.480e+02,
                          3.490e+02, 3.500e+02, 3.510e+02, 3.520e+02, 3.530e+02, 3.540e+02,
                          3.550e+02, 3.560e+02, 3.570e+02, 3.580e+02, 3.590e+02, 3.600e+02,
                          3.610e+02, 3.620e+02, 3.630e+02, 3.640e+02, 3.650e+02, 3.660e+02,
                          3.670e+02, 3.680e+02, 3.690e+02, 3.700e+02, 3.710e+02, 3.720e+02,
                          3.730e+02, 3.740e+02, 3.750e+02, 3.760e+02, 3.770e+02, 3.780e+02,
                          3.790e+02, 3.800e+02, 3.810e+02, 3.820e+02, 3.830e+02, 3.840e+02,
                          3.850e+02, 3.860e+02, 3.870e+02, 3.880e+02, 3.890e+02, 3.900e+02,
                          3.910e+02, 3.920e+02, 3.930e+02, 3.940e+02, 3.950e+02, 3.960e+02,
                          3.970e+02, 3.980e+02, 3.990e+02, 4.000e+02, 4.010e+02, 4.020e+02,
                          4.030e+02, 4.040e+02, 4.050e+02, 4.060e+02, 4.070e+02, 4.080e+02,
                          4.090e+02, 4.100e+02, 4.110e+02, 4.120e+02, 4.130e+02, 4.140e+02,
                          4.150e+02, 4.160e+02, 4.170e+02, 4.180e+02, 4.190e+02, 4.200e+02,
                          4.210e+02, 4.220e+02, 4.230e+02, 4.240e+02, 4.250e+02, 4.260e+02,
                          4.270e+02, 4.280e+02, 4.290e+02, 4.300e+02, 4.310e+02, 4.320e+02,
                          4.330e+02, 4.340e+02, 4.350e+02, 4.360e+02, 4.370e+02, 4.380e+02,
                          4.390e+02, 4.400e+02, 4.410e+02, 4.420e+02, 4.430e+02, 4.440e+02,
                          4.450e+02, 4.460e+02, 4.470e+02, 4.480e+02, 4.490e+02, 4.500e+02,
                          4.510e+02, 4.520e+02, 4.530e+02, 4.540e+02, 4.550e+02, 4.560e+02,
                          4.570e+02, 4.580e+02, 4.590e+02, 4.600e+02, 4.610e+02, 4.620e+02,
                          4.630e+02, 4.640e+02, 4.650e+02, 4.660e+02, 4.670e+02, 4.680e+02,
                          4.690e+02, 4.700e+02, 4.710e+02, 4.720e+02, 4.730e+02, 4.740e+02,
                          4.750e+02, 4.760e+02, 4.770e+02, 4.780e+02, 4.790e+02, 4.800e+02,
                          4.810e+02, 4.820e+02, 4.830e+02, 4.840e+02, 4.850e+02, 4.860e+02,
                          4.870e+02, 4.880e+02, 4.890e+02, 4.900e+02, 4.910e+02, 4.920e+02,
                          4.930e+02, 4.940e+02, 4.950e+02, 4.960e+02, 4.970e+02, 4.980e+02,
                          4.990e+02, 5.000e+02, 5.010e+02, 5.020e+02, 5.030e+02, 5.040e+02,
                          5.050e+02, 5.060e+02, 5.070e+02, 5.080e+02, 5.090e+02, 5.100e+02,
                          5.110e+02, 5.120e+02, 5.130e+02, 5.140e+02, 5.150e+02, 5.160e+02,
                          5.170e+02, 5.180e+02, 5.190e+02, 5.200e+02, 5.210e+02, 5.220e+02,
                          5.230e+02, 5.240e+02, 5.250e+02, 5.260e+02, 5.270e+02, 5.280e+02,
                          5.290e+02, 5.300e+02, 5.310e+02, 5.320e+02, 5.330e+02, 5.340e+02,
                          5.350e+02, 5.360e+02, 5.370e+02, 5.380e+02, 5.390e+02, 5.400e+02,
                          5.410e+02, 5.420e+02, 5.430e+02, 5.440e+02, 5.450e+02, 5.460e+02,
                          5.470e+02, 5.480e+02, 5.490e+02, 5.500e+02, 5.510e+02, 5.520e+02,
                          5.530e+02, 5.540e+02, 5.550e+02, 5.560e+02, 5.570e+02, 5.580e+02,
                          5.590e+02, 5.600e+02, 5.610e+02, 5.620e+02, 5.630e+02, 5.640e+02,
                          5.650e+02, 5.660e+02, 5.670e+02, 5.680e+02, 5.690e+02, 5.700e+02,
                          5.710e+02, 5.720e+02, 5.730e+02, 5.740e+02, 5.750e+02, 5.760e+02,
                          5.770e+02, 5.780e+02, 5.790e+02, 5.800e+02, 5.810e+02, 5.820e+02,
                          5.830e+02, 5.840e+02, 5.850e+02, 5.860e+02, 5.870e+02, 5.880e+02,
                          5.890e+02, 5.900e+02, 5.910e+02, 5.920e+02, 5.930e+02, 5.940e+02,
                          5.950e+02, 5.960e+02, 5.970e+02, 5.980e+02, 5.990e+02, 6.000e+02,
                          6.010e+02, 6.020e+02, 6.030e+02, 6.040e+02, 6.050e+02, 6.060e+02,
                          6.070e+02, 6.080e+02, 6.090e+02, 6.100e+02, 6.110e+02, 6.120e+02,
                          6.130e+02, 6.140e+02, 6.150e+02, 6.160e+02, 6.170e+02, 6.180e+02,
                          6.190e+02, 6.200e+02, 6.210e+02, 6.220e+02, 6.230e+02, 6.240e+02,
                          6.250e+02, 6.260e+02, 6.270e+02, 6.280e+02, 6.290e+02, 6.300e+02,
                          6.310e+02, 6.320e+02, 6.330e+02, 6.340e+02, 6.350e+02, 6.360e+02,
                          6.370e+02, 6.380e+02, 6.390e+02, 6.400e+02, 6.410e+02, 6.420e+02,
                          6.430e+02, 6.440e+02, 6.450e+02, 6.460e+02, 6.470e+02, 6.480e+02,
                          6.490e+02, 6.500e+02, 6.510e+02, 6.520e+02, 6.530e+02, 6.540e+02,
                          6.550e+02, 6.560e+02, 6.570e+02, 6.580e+02, 6.590e+02, 6.600e+02,
                          6.610e+02, 6.620e+02, 6.630e+02, 6.640e+02, 6.650e+02, 6.660e+02,
                          6.670e+02, 6.680e+02, 6.690e+02, 6.700e+02, 6.710e+02, 6.720e+02,
                          6.730e+02, 6.740e+02, 6.750e+02, 6.760e+02, 6.770e+02, 6.780e+02,
                          6.790e+02, 6.800e+02, 6.810e+02, 6.820e+02, 6.830e+02, 6.840e+02,
                          6.850e+02, 6.860e+02, 6.870e+02, 6.880e+02, 6.890e+02, 6.900e+02,
                          6.910e+02, 6.920e+02, 6.930e+02, 6.940e+02, 6.950e+02, 6.960e+02,
                          6.970e+02, 6.980e+02, 6.990e+02, 7.000e+02, 7.010e+02, 7.020e+02,
                          7.030e+02, 7.040e+02, 7.050e+02, 7.060e+02, 7.070e+02, 7.080e+02,
                          7.090e+02, 7.100e+02, 7.110e+02, 7.120e+02, 7.130e+02, 7.140e+02,
                          7.150e+02, 7.160e+02, 7.170e+02, 7.180e+02, 7.190e+02, 7.200e+02,
                          7.210e+02, 7.220e+02, 7.230e+02, 7.240e+02, 7.250e+02, 7.260e+02,
                          7.270e+02, 7.280e+02, 7.290e+02, 7.300e+02, 7.310e+02, 7.320e+02,
                          7.330e+02, 7.340e+02, 7.350e+02, 7.360e+02, 7.370e+02, 7.380e+02,
                          7.390e+02, 7.400e+02, 7.410e+02, 7.420e+02, 7.430e+02, 7.440e+02,
                          7.450e+02, 7.460e+02, 7.470e+02, 7.480e+02, 7.490e+02, 7.500e+02,
                          7.510e+02, 7.520e+02, 7.530e+02, 7.540e+02, 7.550e+02, 7.560e+02,
                          7.570e+02, 7.580e+02, 7.590e+02, 7.600e+02, 7.610e+02, 7.620e+02,
                          7.630e+02, 7.640e+02, 7.650e+02, 7.660e+02, 7.670e+02, 7.680e+02,
                          7.690e+02, 7.700e+02, 7.710e+02, 7.720e+02, 7.730e+02, 7.740e+02,
                          7.750e+02, 7.760e+02, 7.770e+02, 7.780e+02, 7.790e+02, 7.800e+02,
                          7.810e+02, 7.820e+02, 7.830e+02, 7.840e+02, 7.850e+02, 7.860e+02,
                          7.870e+02, 7.880e+02, 7.890e+02, 7.900e+02, 7.910e+02, 7.920e+02,
                          7.930e+02, 7.940e+02, 7.950e+02, 7.960e+02, 7.970e+02, 7.980e+02,
                          7.990e+02, 8.000e+02, 8.010e+02, 8.020e+02, 8.030e+02, 8.040e+02,
                          8.050e+02, 8.060e+02, 8.070e+02, 8.080e+02, 8.090e+02, 8.100e+02,
                          8.110e+02, 8.120e+02, 8.130e+02, 8.140e+02, 8.150e+02, 8.160e+02,
                          8.170e+02, 8.180e+02, 8.190e+02, 8.200e+02, 8.210e+02, 8.220e+02,
                          8.230e+02, 8.240e+02, 8.250e+02, 8.260e+02, 8.270e+02, 8.280e+02,
                          8.290e+02, 8.300e+02, 8.310e+02, 8.320e+02, 8.330e+02, 8.340e+02,
                          8.350e+02, 8.360e+02, 8.370e+02, 8.380e+02, 8.390e+02, 8.400e+02,
                          8.410e+02, 8.420e+02, 8.430e+02, 8.440e+02, 8.450e+02, 8.460e+02,
                          8.470e+02, 8.480e+02, 8.490e+02, 8.500e+02, 8.510e+02, 8.520e+02,
                          8.530e+02, 8.540e+02, 8.550e+02, 8.560e+02, 8.570e+02, 8.580e+02,
                          8.590e+02, 8.600e+02, 8.610e+02, 8.620e+02, 8.630e+02, 8.640e+02,
                          8.650e+02, 8.660e+02, 8.670e+02, 8.680e+02, 8.690e+02, 8.700e+02,
                          8.710e+02, 8.720e+02, 8.730e+02, 8.740e+02, 8.750e+02, 8.760e+02,
                          8.770e+02, 8.780e+02, 8.790e+02, 8.800e+02, 8.810e+02, 8.820e+02,
                          8.830e+02, 8.840e+02, 8.850e+02, 8.860e+02, 8.870e+02, 8.880e+02,
                          8.890e+02, 8.900e+02, 8.910e+02, 8.920e+02, 8.930e+02, 8.940e+02,
                          8.950e+02, 8.960e+02, 8.970e+02, 8.980e+02, 8.990e+02, 9.000e+02,
                          9.010e+02, 9.020e+02, 9.030e+02, 9.040e+02, 9.050e+02, 9.060e+02,
                          9.070e+02, 9.080e+02, 9.090e+02, 9.100e+02, 9.110e+02, 9.120e+02,
                          9.130e+02, 9.140e+02, 9.150e+02, 9.160e+02, 9.170e+02, 9.180e+02,
                          9.190e+02, 9.200e+02, 9.210e+02, 9.220e+02, 9.230e+02, 9.240e+02,
                          9.250e+02, 9.260e+02, 9.270e+02, 9.280e+02, 9.290e+02, 9.300e+02,
                          9.310e+02, 9.320e+02, 9.330e+02, 9.340e+02, 9.350e+02, 9.360e+02,
                          9.370e+02, 9.380e+02, 9.390e+02, 9.400e+02, 9.410e+02, 9.420e+02,
                          9.430e+02, 9.440e+02, 9.450e+02, 9.460e+02, 9.470e+02, 9.480e+02,
                          9.490e+02, 9.500e+02, 9.510e+02, 9.520e+02, 9.530e+02, 9.540e+02,
                          9.550e+02, 9.560e+02, 9.570e+02, 9.580e+02, 9.590e+02, 9.600e+02,
                          9.610e+02, 9.620e+02, 9.630e+02, 9.640e+02, 9.650e+02, 9.660e+02,
                          9.670e+02, 9.680e+02, 9.690e+02, 9.700e+02, 9.710e+02, 9.720e+02,
                          9.730e+02, 9.740e+02, 9.750e+02, 9.760e+02, 9.770e+02, 9.780e+02,
                          9.790e+02, 9.800e+02, 9.810e+02, 9.820e+02, 9.830e+02, 9.840e+02,
                          9.850e+02, 9.860e+02, 9.870e+02, 9.880e+02, 9.890e+02, 9.900e+02,
                          9.910e+02, 9.920e+02, 9.930e+02, 9.940e+02, 9.950e+02, 9.960e+02,
                          9.970e+02, 9.980e+02, 9.990e+02, 1.000e+03, 1.001e+03, 1.002e+03,
                          1.003e+03, 1.004e+03, 1.005e+03, 1.006e+03, 1.007e+03, 1.008e+03,
                          1.009e+03, 1.010e+03, 1.011e+03, 1.012e+03, 1.013e+03, 1.014e+03,
                          1.015e+03, 1.016e+03, 1.017e+03, 1.018e+03, 1.019e+03, 1.020e+03,
                          1.021e+03, 1.022e+03, 1.023e+03, 1.024e+03};
  float CoreB[32 * 32] = {0.000e+00, 2.000e+00, 4.000e+00, 6.000e+00, 8.000e+00, 1.000e+01,
                          1.200e+01, 1.400e+01, 1.600e+01, 1.800e+01, 2.000e+01, 2.200e+01,
                          2.400e+01, 2.600e+01, 2.800e+01, 3.000e+01, 3.200e+01, 3.400e+01,
                          3.600e+01, 3.800e+01, 4.000e+01, 4.200e+01, 4.400e+01, 4.600e+01,
                          4.800e+01, 5.000e+01, 5.200e+01, 5.400e+01, 5.600e+01, 5.800e+01,
                          6.000e+01, 6.200e+01, 6.400e+01, 6.600e+01, 6.800e+01, 7.000e+01,
                          7.200e+01, 7.400e+01, 7.600e+01, 7.800e+01, 8.000e+01, 8.200e+01,
                          8.400e+01, 8.600e+01, 8.800e+01, 9.000e+01, 9.200e+01, 9.400e+01,
                          9.600e+01, 9.800e+01, 1.000e+02, 1.020e+02, 1.040e+02, 1.060e+02,
                          1.080e+02, 1.100e+02, 1.120e+02, 1.140e+02, 1.160e+02, 1.180e+02,
                          1.200e+02, 1.220e+02, 1.240e+02, 1.260e+02, 1.280e+02, 1.300e+02,
                          1.320e+02, 1.340e+02, 1.360e+02, 1.380e+02, 1.400e+02, 1.420e+02,
                          1.440e+02, 1.460e+02, 1.480e+02, 1.500e+02, 1.520e+02, 1.540e+02,
                          1.560e+02, 1.580e+02, 1.600e+02, 1.620e+02, 1.640e+02, 1.660e+02,
                          1.680e+02, 1.700e+02, 1.720e+02, 1.740e+02, 1.760e+02, 1.780e+02,
                          1.800e+02, 1.820e+02, 1.840e+02, 1.860e+02, 1.880e+02, 1.900e+02,
                          1.920e+02, 1.940e+02, 1.960e+02, 1.980e+02, 2.000e+02, 2.020e+02,
                          2.040e+02, 2.060e+02, 2.080e+02, 2.100e+02, 2.120e+02, 2.140e+02,
                          2.160e+02, 2.180e+02, 2.200e+02, 2.220e+02, 2.240e+02, 2.260e+02,
                          2.280e+02, 2.300e+02, 2.320e+02, 2.340e+02, 2.360e+02, 2.380e+02,
                          2.400e+02, 2.420e+02, 2.440e+02, 2.460e+02, 2.480e+02, 2.500e+02,
                          2.520e+02, 2.540e+02, 2.560e+02, 2.580e+02, 2.600e+02, 2.620e+02,
                          2.640e+02, 2.660e+02, 2.680e+02, 2.700e+02, 2.720e+02, 2.740e+02,
                          2.760e+02, 2.780e+02, 2.800e+02, 2.820e+02, 2.840e+02, 2.860e+02,
                          2.880e+02, 2.900e+02, 2.920e+02, 2.940e+02, 2.960e+02, 2.980e+02,
                          3.000e+02, 3.020e+02, 3.040e+02, 3.060e+02, 3.080e+02, 3.100e+02,
                          3.120e+02, 3.140e+02, 3.160e+02, 3.180e+02, 3.200e+02, 3.220e+02,
                          3.240e+02, 3.260e+02, 3.280e+02, 3.300e+02, 3.320e+02, 3.340e+02,
                          3.360e+02, 3.380e+02, 3.400e+02, 3.420e+02, 3.440e+02, 3.460e+02,
                          3.480e+02, 3.500e+02, 3.520e+02, 3.540e+02, 3.560e+02, 3.580e+02,
                          3.600e+02, 3.620e+02, 3.640e+02, 3.660e+02, 3.680e+02, 3.700e+02,
                          3.720e+02, 3.740e+02, 3.760e+02, 3.780e+02, 3.800e+02, 3.820e+02,
                          3.840e+02, 3.860e+02, 3.880e+02, 3.900e+02, 3.920e+02, 3.940e+02,
                          3.960e+02, 3.980e+02, 4.000e+02, 4.020e+02, 4.040e+02, 4.060e+02,
                          4.080e+02, 4.100e+02, 4.120e+02, 4.140e+02, 4.160e+02, 4.180e+02,
                          4.200e+02, 4.220e+02, 4.240e+02, 4.260e+02, 4.280e+02, 4.300e+02,
                          4.320e+02, 4.340e+02, 4.360e+02, 4.380e+02, 4.400e+02, 4.420e+02,
                          4.440e+02, 4.460e+02, 4.480e+02, 4.500e+02, 4.520e+02, 4.540e+02,
                          4.560e+02, 4.580e+02, 4.600e+02, 4.620e+02, 4.640e+02, 4.660e+02,
                          4.680e+02, 4.700e+02, 4.720e+02, 4.740e+02, 4.760e+02, 4.780e+02,
                          4.800e+02, 4.820e+02, 4.840e+02, 4.860e+02, 4.880e+02, 4.900e+02,
                          4.920e+02, 4.940e+02, 4.960e+02, 4.980e+02, 5.000e+02, 5.020e+02,
                          5.040e+02, 5.060e+02, 5.080e+02, 5.100e+02, 5.120e+02, 5.140e+02,
                          5.160e+02, 5.180e+02, 5.200e+02, 5.220e+02, 5.240e+02, 5.260e+02,
                          5.280e+02, 5.300e+02, 5.320e+02, 5.340e+02, 5.360e+02, 5.380e+02,
                          5.400e+02, 5.420e+02, 5.440e+02, 5.460e+02, 5.480e+02, 5.500e+02,
                          5.520e+02, 5.540e+02, 5.560e+02, 5.580e+02, 5.600e+02, 5.620e+02,
                          5.640e+02, 5.660e+02, 5.680e+02, 5.700e+02, 5.720e+02, 5.740e+02,
                          5.760e+02, 5.780e+02, 5.800e+02, 5.820e+02, 5.840e+02, 5.860e+02,
                          5.880e+02, 5.900e+02, 5.920e+02, 5.940e+02, 5.960e+02, 5.980e+02,
                          6.000e+02, 6.020e+02, 6.040e+02, 6.060e+02, 6.080e+02, 6.100e+02,
                          6.120e+02, 6.140e+02, 6.160e+02, 6.180e+02, 6.200e+02, 6.220e+02,
                          6.240e+02, 6.260e+02, 6.280e+02, 6.300e+02, 6.320e+02, 6.340e+02,
                          6.360e+02, 6.380e+02, 6.400e+02, 6.420e+02, 6.440e+02, 6.460e+02,
                          6.480e+02, 6.500e+02, 6.520e+02, 6.540e+02, 6.560e+02, 6.580e+02,
                          6.600e+02, 6.620e+02, 6.640e+02, 6.660e+02, 6.680e+02, 6.700e+02,
                          6.720e+02, 6.740e+02, 6.760e+02, 6.780e+02, 6.800e+02, 6.820e+02,
                          6.840e+02, 6.860e+02, 6.880e+02, 6.900e+02, 6.920e+02, 6.940e+02,
                          6.960e+02, 6.980e+02, 7.000e+02, 7.020e+02, 7.040e+02, 7.060e+02,
                          7.080e+02, 7.100e+02, 7.120e+02, 7.140e+02, 7.160e+02, 7.180e+02,
                          7.200e+02, 7.220e+02, 7.240e+02, 7.260e+02, 7.280e+02, 7.300e+02,
                          7.320e+02, 7.340e+02, 7.360e+02, 7.380e+02, 7.400e+02, 7.420e+02,
                          7.440e+02, 7.460e+02, 7.480e+02, 7.500e+02, 7.520e+02, 7.540e+02,
                          7.560e+02, 7.580e+02, 7.600e+02, 7.620e+02, 7.640e+02, 7.660e+02,
                          7.680e+02, 7.700e+02, 7.720e+02, 7.740e+02, 7.760e+02, 7.780e+02,
                          7.800e+02, 7.820e+02, 7.840e+02, 7.860e+02, 7.880e+02, 7.900e+02,
                          7.920e+02, 7.940e+02, 7.960e+02, 7.980e+02, 8.000e+02, 8.020e+02,
                          8.040e+02, 8.060e+02, 8.080e+02, 8.100e+02, 8.120e+02, 8.140e+02,
                          8.160e+02, 8.180e+02, 8.200e+02, 8.220e+02, 8.240e+02, 8.260e+02,
                          8.280e+02, 8.300e+02, 8.320e+02, 8.340e+02, 8.360e+02, 8.380e+02,
                          8.400e+02, 8.420e+02, 8.440e+02, 8.460e+02, 8.480e+02, 8.500e+02,
                          8.520e+02, 8.540e+02, 8.560e+02, 8.580e+02, 8.600e+02, 8.620e+02,
                          8.640e+02, 8.660e+02, 8.680e+02, 8.700e+02, 8.720e+02, 8.740e+02,
                          8.760e+02, 8.780e+02, 8.800e+02, 8.820e+02, 8.840e+02, 8.860e+02,
                          8.880e+02, 8.900e+02, 8.920e+02, 8.940e+02, 8.960e+02, 8.980e+02,
                          9.000e+02, 9.020e+02, 9.040e+02, 9.060e+02, 9.080e+02, 9.100e+02,
                          9.120e+02, 9.140e+02, 9.160e+02, 9.180e+02, 9.200e+02, 9.220e+02,
                          9.240e+02, 9.260e+02, 9.280e+02, 9.300e+02, 9.320e+02, 9.340e+02,
                          9.360e+02, 9.380e+02, 9.400e+02, 9.420e+02, 9.440e+02, 9.460e+02,
                          9.480e+02, 9.500e+02, 9.520e+02, 9.540e+02, 9.560e+02, 9.580e+02,
                          9.600e+02, 9.620e+02, 9.640e+02, 9.660e+02, 9.680e+02, 9.700e+02,
                          9.720e+02, 9.740e+02, 9.760e+02, 9.780e+02, 9.800e+02, 9.820e+02,
                          9.840e+02, 9.860e+02, 9.880e+02, 9.900e+02, 9.920e+02, 9.940e+02,
                          9.960e+02, 9.980e+02, 1.000e+03, 1.002e+03, 1.004e+03, 1.006e+03,
                          1.008e+03, 1.010e+03, 1.012e+03, 1.014e+03, 1.016e+03, 1.018e+03,
                          1.020e+03, 1.022e+03, 1.024e+03, 1.026e+03, 1.028e+03, 1.030e+03,
                          1.032e+03, 1.034e+03, 1.036e+03, 1.038e+03, 1.040e+03, 1.042e+03,
                          1.044e+03, 1.046e+03, 1.048e+03, 1.050e+03, 1.052e+03, 1.054e+03,
                          1.056e+03, 1.058e+03, 1.060e+03, 1.062e+03, 1.064e+03, 1.066e+03,
                          1.068e+03, 1.070e+03, 1.072e+03, 1.074e+03, 1.076e+03, 1.078e+03,
                          1.080e+03, 1.082e+03, 1.084e+03, 1.086e+03, 1.088e+03, 1.090e+03,
                          1.092e+03, 1.094e+03, 1.096e+03, 1.098e+03, 1.100e+03, 1.102e+03,
                          1.104e+03, 1.106e+03, 1.108e+03, 1.110e+03, 1.112e+03, 1.114e+03,
                          1.116e+03, 1.118e+03, 1.120e+03, 1.122e+03, 1.124e+03, 1.126e+03,
                          1.128e+03, 1.130e+03, 1.132e+03, 1.134e+03, 1.136e+03, 1.138e+03,
                          1.140e+03, 1.142e+03, 1.144e+03, 1.146e+03, 1.148e+03, 1.150e+03,
                          1.152e+03, 1.154e+03, 1.156e+03, 1.158e+03, 1.160e+03, 1.162e+03,
                          1.164e+03, 1.166e+03, 1.168e+03, 1.170e+03, 1.172e+03, 1.174e+03,
                          1.176e+03, 1.178e+03, 1.180e+03, 1.182e+03, 1.184e+03, 1.186e+03,
                          1.188e+03, 1.190e+03, 1.192e+03, 1.194e+03, 1.196e+03, 1.198e+03,
                          1.200e+03, 1.202e+03, 1.204e+03, 1.206e+03, 1.208e+03, 1.210e+03,
                          1.212e+03, 1.214e+03, 1.216e+03, 1.218e+03, 1.220e+03, 1.222e+03,
                          1.224e+03, 1.226e+03, 1.228e+03, 1.230e+03, 1.232e+03, 1.234e+03,
                          1.236e+03, 1.238e+03, 1.240e+03, 1.242e+03, 1.244e+03, 1.246e+03,
                          1.248e+03, 1.250e+03, 1.252e+03, 1.254e+03, 1.256e+03, 1.258e+03,
                          1.260e+03, 1.262e+03, 1.264e+03, 1.266e+03, 1.268e+03, 1.270e+03,
                          1.272e+03, 1.274e+03, 1.276e+03, 1.278e+03, 1.280e+03, 1.282e+03,
                          1.284e+03, 1.286e+03, 1.288e+03, 1.290e+03, 1.292e+03, 1.294e+03,
                          1.296e+03, 1.298e+03, 1.300e+03, 1.302e+03, 1.304e+03, 1.306e+03,
                          1.308e+03, 1.310e+03, 1.312e+03, 1.314e+03, 1.316e+03, 1.318e+03,
                          1.320e+03, 1.322e+03, 1.324e+03, 1.326e+03, 1.328e+03, 1.330e+03,
                          1.332e+03, 1.334e+03, 1.336e+03, 1.338e+03, 1.340e+03, 1.342e+03,
                          1.344e+03, 1.346e+03, 1.348e+03, 1.350e+03, 1.352e+03, 1.354e+03,
                          1.356e+03, 1.358e+03, 1.360e+03, 1.362e+03, 1.364e+03, 1.366e+03,
                          1.368e+03, 1.370e+03, 1.372e+03, 1.374e+03, 1.376e+03, 1.378e+03,
                          1.380e+03, 1.382e+03, 1.384e+03, 1.386e+03, 1.388e+03, 1.390e+03,
                          1.392e+03, 1.394e+03, 1.396e+03, 1.398e+03, 1.400e+03, 1.402e+03,
                          1.404e+03, 1.406e+03, 1.408e+03, 1.410e+03, 1.412e+03, 1.414e+03,
                          1.416e+03, 1.418e+03, 1.420e+03, 1.422e+03, 1.424e+03, 1.426e+03,
                          1.428e+03, 1.430e+03, 1.432e+03, 1.434e+03, 1.436e+03, 1.438e+03,
                          1.440e+03, 1.442e+03, 1.444e+03, 1.446e+03, 1.448e+03, 1.450e+03,
                          1.452e+03, 1.454e+03, 1.456e+03, 1.458e+03, 1.460e+03, 1.462e+03,
                          1.464e+03, 1.466e+03, 1.468e+03, 1.470e+03, 1.472e+03, 1.474e+03,
                          1.476e+03, 1.478e+03, 1.480e+03, 1.482e+03, 1.484e+03, 1.486e+03,
                          1.488e+03, 1.490e+03, 1.492e+03, 1.494e+03, 1.496e+03, 1.498e+03,
                          1.500e+03, 1.502e+03, 1.504e+03, 1.506e+03, 1.508e+03, 1.510e+03,
                          1.512e+03, 1.514e+03, 1.516e+03, 1.518e+03, 1.520e+03, 1.522e+03,
                          1.524e+03, 1.526e+03, 1.528e+03, 1.530e+03, 1.532e+03, 1.534e+03,
                          1.536e+03, 1.538e+03, 1.540e+03, 1.542e+03, 1.544e+03, 1.546e+03,
                          1.548e+03, 1.550e+03, 1.552e+03, 1.554e+03, 1.556e+03, 1.558e+03,
                          1.560e+03, 1.562e+03, 1.564e+03, 1.566e+03, 1.568e+03, 1.570e+03,
                          1.572e+03, 1.574e+03, 1.576e+03, 1.578e+03, 1.580e+03, 1.582e+03,
                          1.584e+03, 1.586e+03, 1.588e+03, 1.590e+03, 1.592e+03, 1.594e+03,
                          1.596e+03, 1.598e+03, 1.600e+03, 1.602e+03, 1.604e+03, 1.606e+03,
                          1.608e+03, 1.610e+03, 1.612e+03, 1.614e+03, 1.616e+03, 1.618e+03,
                          1.620e+03, 1.622e+03, 1.624e+03, 1.626e+03, 1.628e+03, 1.630e+03,
                          1.632e+03, 1.634e+03, 1.636e+03, 1.638e+03, 1.640e+03, 1.642e+03,
                          1.644e+03, 1.646e+03, 1.648e+03, 1.650e+03, 1.652e+03, 1.654e+03,
                          1.656e+03, 1.658e+03, 1.660e+03, 1.662e+03, 1.664e+03, 1.666e+03,
                          1.668e+03, 1.670e+03, 1.672e+03, 1.674e+03, 1.676e+03, 1.678e+03,
                          1.680e+03, 1.682e+03, 1.684e+03, 1.686e+03, 1.688e+03, 1.690e+03,
                          1.692e+03, 1.694e+03, 1.696e+03, 1.698e+03, 1.700e+03, 1.702e+03,
                          1.704e+03, 1.706e+03, 1.708e+03, 1.710e+03, 1.712e+03, 1.714e+03,
                          1.716e+03, 1.718e+03, 1.720e+03, 1.722e+03, 1.724e+03, 1.726e+03,
                          1.728e+03, 1.730e+03, 1.732e+03, 1.734e+03, 1.736e+03, 1.738e+03,
                          1.740e+03, 1.742e+03, 1.744e+03, 1.746e+03, 1.748e+03, 1.750e+03,
                          1.752e+03, 1.754e+03, 1.756e+03, 1.758e+03, 1.760e+03, 1.762e+03,
                          1.764e+03, 1.766e+03, 1.768e+03, 1.770e+03, 1.772e+03, 1.774e+03,
                          1.776e+03, 1.778e+03, 1.780e+03, 1.782e+03, 1.784e+03, 1.786e+03,
                          1.788e+03, 1.790e+03, 1.792e+03, 1.794e+03, 1.796e+03, 1.798e+03,
                          1.800e+03, 1.802e+03, 1.804e+03, 1.806e+03, 1.808e+03, 1.810e+03,
                          1.812e+03, 1.814e+03, 1.816e+03, 1.818e+03, 1.820e+03, 1.822e+03,
                          1.824e+03, 1.826e+03, 1.828e+03, 1.830e+03, 1.832e+03, 1.834e+03,
                          1.836e+03, 1.838e+03, 1.840e+03, 1.842e+03, 1.844e+03, 1.846e+03,
                          1.848e+03, 1.850e+03, 1.852e+03, 1.854e+03, 1.856e+03, 1.858e+03,
                          1.860e+03, 1.862e+03, 1.864e+03, 1.866e+03, 1.868e+03, 1.870e+03,
                          1.872e+03, 1.874e+03, 1.876e+03, 1.878e+03, 1.880e+03, 1.882e+03,
                          1.884e+03, 1.886e+03, 1.888e+03, 1.890e+03, 1.892e+03, 1.894e+03,
                          1.896e+03, 1.898e+03, 1.900e+03, 1.902e+03, 1.904e+03, 1.906e+03,
                          1.908e+03, 1.910e+03, 1.912e+03, 1.914e+03, 1.916e+03, 1.918e+03,
                          1.920e+03, 1.922e+03, 1.924e+03, 1.926e+03, 1.928e+03, 1.930e+03,
                          1.932e+03, 1.934e+03, 1.936e+03, 1.938e+03, 1.940e+03, 1.942e+03,
                          1.944e+03, 1.946e+03, 1.948e+03, 1.950e+03, 1.952e+03, 1.954e+03,
                          1.956e+03, 1.958e+03, 1.960e+03, 1.962e+03, 1.964e+03, 1.966e+03,
                          1.968e+03, 1.970e+03, 1.972e+03, 1.974e+03, 1.976e+03, 1.978e+03,
                          1.980e+03, 1.982e+03, 1.984e+03, 1.986e+03, 1.988e+03, 1.990e+03,
                          1.992e+03, 1.994e+03, 1.996e+03, 1.998e+03, 2.000e+03, 2.002e+03,
                          2.004e+03, 2.006e+03, 2.008e+03, 2.010e+03, 2.012e+03, 2.014e+03,
                          2.016e+03, 2.018e+03, 2.020e+03, 2.022e+03, 2.024e+03, 2.026e+03,
                          2.028e+03, 2.030e+03, 2.032e+03, 2.034e+03, 2.036e+03, 2.038e+03,
                          2.040e+03, 2.042e+03, 2.044e+03, 2.046e+03};
  float CoreC[32 * 32] = {0.000e+00, 1.000e-01, 2.000e-01, 3.000e-01, 4.000e-01, 5.000e-01,
                          6.000e-01, 7.000e-01, 8.000e-01, 9.000e-01, 1.000e+00, 1.100e+00,
                          1.200e+00, 1.300e+00, 1.400e+00, 1.500e+00, 1.600e+00, 1.700e+00,
                          1.800e+00, 1.900e+00, 2.000e+00, 2.100e+00, 2.200e+00, 2.300e+00,
                          2.400e+00, 2.500e+00, 2.600e+00, 2.700e+00, 2.800e+00, 2.900e+00,
                          3.000e+00, 3.100e+00, 3.200e+00, 3.300e+00, 3.400e+00, 3.500e+00,
                          3.600e+00, 3.700e+00, 3.800e+00, 3.900e+00, 4.000e+00, 4.100e+00,
                          4.200e+00, 4.300e+00, 4.400e+00, 4.500e+00, 4.600e+00, 4.700e+00,
                          4.800e+00, 4.900e+00, 5.000e+00, 5.100e+00, 5.200e+00, 5.300e+00,
                          5.400e+00, 5.500e+00, 5.600e+00, 5.700e+00, 5.800e+00, 5.900e+00,
                          6.000e+00, 6.100e+00, 6.200e+00, 6.300e+00, 6.400e+00, 6.500e+00,
                          6.600e+00, 6.700e+00, 6.800e+00, 6.900e+00, 7.000e+00, 7.100e+00,
                          7.200e+00, 7.300e+00, 7.400e+00, 7.500e+00, 7.600e+00, 7.700e+00,
                          7.800e+00, 7.900e+00, 8.000e+00, 8.100e+00, 8.200e+00, 8.300e+00,
                          8.400e+00, 8.500e+00, 8.600e+00, 8.700e+00, 8.800e+00, 8.900e+00,
                          9.000e+00, 9.100e+00, 9.200e+00, 9.300e+00, 9.400e+00, 9.500e+00,
                          9.600e+00, 9.700e+00, 9.800e+00, 9.900e+00, 1.000e+01, 1.010e+01,
                          1.020e+01, 1.030e+01, 1.040e+01, 1.050e+01, 1.060e+01, 1.070e+01,
                          1.080e+01, 1.090e+01, 1.100e+01, 1.110e+01, 1.120e+01, 1.130e+01,
                          1.140e+01, 1.150e+01, 1.160e+01, 1.170e+01, 1.180e+01, 1.190e+01,
                          1.200e+01, 1.210e+01, 1.220e+01, 1.230e+01, 1.240e+01, 1.250e+01,
                          1.260e+01, 1.270e+01, 1.280e+01, 1.290e+01, 1.300e+01, 1.310e+01,
                          1.320e+01, 1.330e+01, 1.340e+01, 1.350e+01, 1.360e+01, 1.370e+01,
                          1.380e+01, 1.390e+01, 1.400e+01, 1.410e+01, 1.420e+01, 1.430e+01,
                          1.440e+01, 1.450e+01, 1.460e+01, 1.470e+01, 1.480e+01, 1.490e+01,
                          1.500e+01, 1.510e+01, 1.520e+01, 1.530e+01, 1.540e+01, 1.550e+01,
                          1.560e+01, 1.570e+01, 1.580e+01, 1.590e+01, 1.600e+01, 1.610e+01,
                          1.620e+01, 1.630e+01, 1.640e+01, 1.650e+01, 1.660e+01, 1.670e+01,
                          1.680e+01, 1.690e+01, 1.700e+01, 1.710e+01, 1.720e+01, 1.730e+01,
                          1.740e+01, 1.750e+01, 1.760e+01, 1.770e+01, 1.780e+01, 1.790e+01,
                          1.800e+01, 1.810e+01, 1.820e+01, 1.830e+01, 1.840e+01, 1.850e+01,
                          1.860e+01, 1.870e+01, 1.880e+01, 1.890e+01, 1.900e+01, 1.910e+01,
                          1.920e+01, 1.930e+01, 1.940e+01, 1.950e+01, 1.960e+01, 1.970e+01,
                          1.980e+01, 1.990e+01, 2.000e+01, 2.010e+01, 2.020e+01, 2.030e+01,
                          2.040e+01, 2.050e+01, 2.060e+01, 2.070e+01, 2.080e+01, 2.090e+01,
                          2.100e+01, 2.110e+01, 2.120e+01, 2.130e+01, 2.140e+01, 2.150e+01,
                          2.160e+01, 2.170e+01, 2.180e+01, 2.190e+01, 2.200e+01, 2.210e+01,
                          2.220e+01, 2.230e+01, 2.240e+01, 2.250e+01, 2.260e+01, 2.270e+01,
                          2.280e+01, 2.290e+01, 2.300e+01, 2.310e+01, 2.320e+01, 2.330e+01,
                          2.340e+01, 2.350e+01, 2.360e+01, 2.370e+01, 2.380e+01, 2.390e+01,
                          2.400e+01, 2.410e+01, 2.420e+01, 2.430e+01, 2.440e+01, 2.450e+01,
                          2.460e+01, 2.470e+01, 2.480e+01, 2.490e+01, 2.500e+01, 2.510e+01,
                          2.520e+01, 2.530e+01, 2.540e+01, 2.550e+01, 2.560e+01, 2.570e+01,
                          2.580e+01, 2.590e+01, 2.600e+01, 2.610e+01, 2.620e+01, 2.630e+01,
                          2.640e+01, 2.650e+01, 2.660e+01, 2.670e+01, 2.680e+01, 2.690e+01,
                          2.700e+01, 2.710e+01, 2.720e+01, 2.730e+01, 2.740e+01, 2.750e+01,
                          2.760e+01, 2.770e+01, 2.780e+01, 2.790e+01, 2.800e+01, 2.810e+01,
                          2.820e+01, 2.830e+01, 2.840e+01, 2.850e+01, 2.860e+01, 2.870e+01,
                          2.880e+01, 2.890e+01, 2.900e+01, 2.910e+01, 2.920e+01, 2.930e+01,
                          2.940e+01, 2.950e+01, 2.960e+01, 2.970e+01, 2.980e+01, 2.990e+01,
                          3.000e+01, 3.010e+01, 3.020e+01, 3.030e+01, 3.040e+01, 3.050e+01,
                          3.060e+01, 3.070e+01, 3.080e+01, 3.090e+01, 3.100e+01, 3.110e+01,
                          3.120e+01, 3.130e+01, 3.140e+01, 3.150e+01, 3.160e+01, 3.170e+01,
                          3.180e+01, 3.190e+01, 3.200e+01, 3.210e+01, 3.220e+01, 3.230e+01,
                          3.240e+01, 3.250e+01, 3.260e+01, 3.270e+01, 3.280e+01, 3.290e+01,
                          3.300e+01, 3.310e+01, 3.320e+01, 3.330e+01, 3.340e+01, 3.350e+01,
                          3.360e+01, 3.370e+01, 3.380e+01, 3.390e+01, 3.400e+01, 3.410e+01,
                          3.420e+01, 3.430e+01, 3.440e+01, 3.450e+01, 3.460e+01, 3.470e+01,
                          3.480e+01, 3.490e+01, 3.500e+01, 3.510e+01, 3.520e+01, 3.530e+01,
                          3.540e+01, 3.550e+01, 3.560e+01, 3.570e+01, 3.580e+01, 3.590e+01,
                          3.600e+01, 3.610e+01, 3.620e+01, 3.630e+01, 3.640e+01, 3.650e+01,
                          3.660e+01, 3.670e+01, 3.680e+01, 3.690e+01, 3.700e+01, 3.710e+01,
                          3.720e+01, 3.730e+01, 3.740e+01, 3.750e+01, 3.760e+01, 3.770e+01,
                          3.780e+01, 3.790e+01, 3.800e+01, 3.810e+01, 3.820e+01, 3.830e+01,
                          3.840e+01, 3.850e+01, 3.860e+01, 3.870e+01, 3.880e+01, 3.890e+01,
                          3.900e+01, 3.910e+01, 3.920e+01, 3.930e+01, 3.940e+01, 3.950e+01,
                          3.960e+01, 3.970e+01, 3.980e+01, 3.990e+01, 4.000e+01, 4.010e+01,
                          4.020e+01, 4.030e+01, 4.040e+01, 4.050e+01, 4.060e+01, 4.070e+01,
                          4.080e+01, 4.090e+01, 4.100e+01, 4.110e+01, 4.120e+01, 4.130e+01,
                          4.140e+01, 4.150e+01, 4.160e+01, 4.170e+01, 4.180e+01, 4.190e+01,
                          4.200e+01, 4.210e+01, 4.220e+01, 4.230e+01, 4.240e+01, 4.250e+01,
                          4.260e+01, 4.270e+01, 4.280e+01, 4.290e+01, 4.300e+01, 4.310e+01,
                          4.320e+01, 4.330e+01, 4.340e+01, 4.350e+01, 4.360e+01, 4.370e+01,
                          4.380e+01, 4.390e+01, 4.400e+01, 4.410e+01, 4.420e+01, 4.430e+01,
                          4.440e+01, 4.450e+01, 4.460e+01, 4.470e+01, 4.480e+01, 4.490e+01,
                          4.500e+01, 4.510e+01, 4.520e+01, 4.530e+01, 4.540e+01, 4.550e+01,
                          4.560e+01, 4.570e+01, 4.580e+01, 4.590e+01, 4.600e+01, 4.610e+01,
                          4.620e+01, 4.630e+01, 4.640e+01, 4.650e+01, 4.660e+01, 4.670e+01,
                          4.680e+01, 4.690e+01, 4.700e+01, 4.710e+01, 4.720e+01, 4.730e+01,
                          4.740e+01, 4.750e+01, 4.760e+01, 4.770e+01, 4.780e+01, 4.790e+01,
                          4.800e+01, 4.810e+01, 4.820e+01, 4.830e+01, 4.840e+01, 4.850e+01,
                          4.860e+01, 4.870e+01, 4.880e+01, 4.890e+01, 4.900e+01, 4.910e+01,
                          4.920e+01, 4.930e+01, 4.940e+01, 4.950e+01, 4.960e+01, 4.970e+01,
                          4.980e+01, 4.990e+01, 5.000e+01, 5.010e+01, 5.020e+01, 5.030e+01,
                          5.040e+01, 5.050e+01, 5.060e+01, 5.070e+01, 5.080e+01, 5.090e+01,
                          5.100e+01, 5.110e+01, 5.120e+01, 5.130e+01, 5.140e+01, 5.150e+01,
                          5.160e+01, 5.170e+01, 5.180e+01, 5.190e+01, 5.200e+01, 5.210e+01,
                          5.220e+01, 5.230e+01, 5.240e+01, 5.250e+01, 5.260e+01, 5.270e+01,
                          5.280e+01, 5.290e+01, 5.300e+01, 5.310e+01, 5.320e+01, 5.330e+01,
                          5.340e+01, 5.350e+01, 5.360e+01, 5.370e+01, 5.380e+01, 5.390e+01,
                          5.400e+01, 5.410e+01, 5.420e+01, 5.430e+01, 5.440e+01, 5.450e+01,
                          5.460e+01, 5.470e+01, 5.480e+01, 5.490e+01, 5.500e+01, 5.510e+01,
                          5.520e+01, 5.530e+01, 5.540e+01, 5.550e+01, 5.560e+01, 5.570e+01,
                          5.580e+01, 5.590e+01, 5.600e+01, 5.610e+01, 5.620e+01, 5.630e+01,
                          5.640e+01, 5.650e+01, 5.660e+01, 5.670e+01, 5.680e+01, 5.690e+01,
                          5.700e+01, 5.710e+01, 5.720e+01, 5.730e+01, 5.740e+01, 5.750e+01,
                          5.760e+01, 5.770e+01, 5.780e+01, 5.790e+01, 5.800e+01, 5.810e+01,
                          5.820e+01, 5.830e+01, 5.840e+01, 5.850e+01, 5.860e+01, 5.870e+01,
                          5.880e+01, 5.890e+01, 5.900e+01, 5.910e+01, 5.920e+01, 5.930e+01,
                          5.940e+01, 5.950e+01, 5.960e+01, 5.970e+01, 5.980e+01, 5.990e+01,
                          6.000e+01, 6.010e+01, 6.020e+01, 6.030e+01, 6.040e+01, 6.050e+01,
                          6.060e+01, 6.070e+01, 6.080e+01, 6.090e+01, 6.100e+01, 6.110e+01,
                          6.120e+01, 6.130e+01, 6.140e+01, 6.150e+01, 6.160e+01, 6.170e+01,
                          6.180e+01, 6.190e+01, 6.200e+01, 6.210e+01, 6.220e+01, 6.230e+01,
                          6.240e+01, 6.250e+01, 6.260e+01, 6.270e+01, 6.280e+01, 6.290e+01,
                          6.300e+01, 6.310e+01, 6.320e+01, 6.330e+01, 6.340e+01, 6.350e+01,
                          6.360e+01, 6.370e+01, 6.380e+01, 6.390e+01, 6.400e+01, 6.410e+01,
                          6.420e+01, 6.430e+01, 6.440e+01, 6.450e+01, 6.460e+01, 6.470e+01,
                          6.480e+01, 6.490e+01, 6.500e+01, 6.510e+01, 6.520e+01, 6.530e+01,
                          6.540e+01, 6.550e+01, 6.560e+01, 6.570e+01, 6.580e+01, 6.590e+01,
                          6.600e+01, 6.610e+01, 6.620e+01, 6.630e+01, 6.640e+01, 6.650e+01,
                          6.660e+01, 6.670e+01, 6.680e+01, 6.690e+01, 6.700e+01, 6.710e+01,
                          6.720e+01, 6.730e+01, 6.740e+01, 6.750e+01, 6.760e+01, 6.770e+01,
                          6.780e+01, 6.790e+01, 6.800e+01, 6.810e+01, 6.820e+01, 6.830e+01,
                          6.840e+01, 6.850e+01, 6.860e+01, 6.870e+01, 6.880e+01, 6.890e+01,
                          6.900e+01, 6.910e+01, 6.920e+01, 6.930e+01, 6.940e+01, 6.950e+01,
                          6.960e+01, 6.970e+01, 6.980e+01, 6.990e+01, 7.000e+01, 7.010e+01,
                          7.020e+01, 7.030e+01, 7.040e+01, 7.050e+01, 7.060e+01, 7.070e+01,
                          7.080e+01, 7.090e+01, 7.100e+01, 7.110e+01, 7.120e+01, 7.130e+01,
                          7.140e+01, 7.150e+01, 7.160e+01, 7.170e+01, 7.180e+01, 7.190e+01,
                          7.200e+01, 7.210e+01, 7.220e+01, 7.230e+01, 7.240e+01, 7.250e+01,
                          7.260e+01, 7.270e+01, 7.280e+01, 7.290e+01, 7.300e+01, 7.310e+01,
                          7.320e+01, 7.330e+01, 7.340e+01, 7.350e+01, 7.360e+01, 7.370e+01,
                          7.380e+01, 7.390e+01, 7.400e+01, 7.410e+01, 7.420e+01, 7.430e+01,
                          7.440e+01, 7.450e+01, 7.460e+01, 7.470e+01, 7.480e+01, 7.490e+01,
                          7.500e+01, 7.510e+01, 7.520e+01, 7.530e+01, 7.540e+01, 7.550e+01,
                          7.560e+01, 7.570e+01, 7.580e+01, 7.590e+01, 7.600e+01, 7.610e+01,
                          7.620e+01, 7.630e+01, 7.640e+01, 7.650e+01, 7.660e+01, 7.670e+01,
                          7.680e+01, 7.690e+01, 7.700e+01, 7.710e+01, 7.720e+01, 7.730e+01,
                          7.740e+01, 7.750e+01, 7.760e+01, 7.770e+01, 7.780e+01, 7.790e+01,
                          7.800e+01, 7.810e+01, 7.820e+01, 7.830e+01, 7.840e+01, 7.850e+01,
                          7.860e+01, 7.870e+01, 7.880e+01, 7.890e+01, 7.900e+01, 7.910e+01,
                          7.920e+01, 7.930e+01, 7.940e+01, 7.950e+01, 7.960e+01, 7.970e+01,
                          7.980e+01, 7.990e+01, 8.000e+01, 8.010e+01, 8.020e+01, 8.030e+01,
                          8.040e+01, 8.050e+01, 8.060e+01, 8.070e+01, 8.080e+01, 8.090e+01,
                          8.100e+01, 8.110e+01, 8.120e+01, 8.130e+01, 8.140e+01, 8.150e+01,
                          8.160e+01, 8.170e+01, 8.180e+01, 8.190e+01, 8.200e+01, 8.210e+01,
                          8.220e+01, 8.230e+01, 8.240e+01, 8.250e+01, 8.260e+01, 8.270e+01,
                          8.280e+01, 8.290e+01, 8.300e+01, 8.310e+01, 8.320e+01, 8.330e+01,
                          8.340e+01, 8.350e+01, 8.360e+01, 8.370e+01, 8.380e+01, 8.390e+01,
                          8.400e+01, 8.410e+01, 8.420e+01, 8.430e+01, 8.440e+01, 8.450e+01,
                          8.460e+01, 8.470e+01, 8.480e+01, 8.490e+01, 8.500e+01, 8.510e+01,
                          8.520e+01, 8.530e+01, 8.540e+01, 8.550e+01, 8.560e+01, 8.570e+01,
                          8.580e+01, 8.590e+01, 8.600e+01, 8.610e+01, 8.620e+01, 8.630e+01,
                          8.640e+01, 8.650e+01, 8.660e+01, 8.670e+01, 8.680e+01, 8.690e+01,
                          8.700e+01, 8.710e+01, 8.720e+01, 8.730e+01, 8.740e+01, 8.750e+01,
                          8.760e+01, 8.770e+01, 8.780e+01, 8.790e+01, 8.800e+01, 8.810e+01,
                          8.820e+01, 8.830e+01, 8.840e+01, 8.850e+01, 8.860e+01, 8.870e+01,
                          8.880e+01, 8.890e+01, 8.900e+01, 8.910e+01, 8.920e+01, 8.930e+01,
                          8.940e+01, 8.950e+01, 8.960e+01, 8.970e+01, 8.980e+01, 8.990e+01,
                          9.000e+01, 9.010e+01, 9.020e+01, 9.030e+01, 9.040e+01, 9.050e+01,
                          9.060e+01, 9.070e+01, 9.080e+01, 9.090e+01, 9.100e+01, 9.110e+01,
                          9.120e+01, 9.130e+01, 9.140e+01, 9.150e+01, 9.160e+01, 9.170e+01,
                          9.180e+01, 9.190e+01, 9.200e+01, 9.210e+01, 9.220e+01, 9.230e+01,
                          9.240e+01, 9.250e+01, 9.260e+01, 9.270e+01, 9.280e+01, 9.290e+01,
                          9.300e+01, 9.310e+01, 9.320e+01, 9.330e+01, 9.340e+01, 9.350e+01,
                          9.360e+01, 9.370e+01, 9.380e+01, 9.390e+01, 9.400e+01, 9.410e+01,
                          9.420e+01, 9.430e+01, 9.440e+01, 9.450e+01, 9.460e+01, 9.470e+01,
                          9.480e+01, 9.490e+01, 9.500e+01, 9.510e+01, 9.520e+01, 9.530e+01,
                          9.540e+01, 9.550e+01, 9.560e+01, 9.570e+01, 9.580e+01, 9.590e+01,
                          9.600e+01, 9.610e+01, 9.620e+01, 9.630e+01, 9.640e+01, 9.650e+01,
                          9.660e+01, 9.670e+01, 9.680e+01, 9.690e+01, 9.700e+01, 9.710e+01,
                          9.720e+01, 9.730e+01, 9.740e+01, 9.750e+01, 9.760e+01, 9.770e+01,
                          9.780e+01, 9.790e+01, 9.800e+01, 9.810e+01, 9.820e+01, 9.830e+01,
                          9.840e+01, 9.850e+01, 9.860e+01, 9.870e+01, 9.880e+01, 9.890e+01,
                          9.900e+01, 9.910e+01, 9.920e+01, 9.930e+01, 9.940e+01, 9.950e+01,
                          9.960e+01, 9.970e+01, 9.980e+01, 9.990e+01, 1.000e+02, 1.001e+02,
                          1.002e+02, 1.003e+02, 1.004e+02, 1.005e+02, 1.006e+02, 1.007e+02,
                          1.008e+02, 1.009e+02, 1.010e+02, 1.011e+02, 1.012e+02, 1.013e+02,
                          1.014e+02, 1.015e+02, 1.016e+02, 1.017e+02, 1.018e+02, 1.019e+02,
                          1.020e+02, 1.021e+02, 1.022e+02, 1.023e+02};

  // Buffers
  std::cout << "Instantiating buffer matrices" << std::endl;
  float *A = new float[32 * 32 * 156728];
  float *B = new float[32 * 32 * 156728];
  float *C = new float[32 * 32 * 156728];
  float *R1 = new float[32 * 32 * 156728];
  float *R2 = new float[32 * 32 * 156728];

  // Copy the Element Matrices N times into Element Buffers
  std::cout << "Copying core matrices to buffers" << std::endl;
  for (int i = 0; i < 156728; i++)
  {
    std::memcpy(&A[32 * 32 * i], &CoreA[0], 32 * 32 * sizeof(float));
    std::memcpy(&B[32 * 32 * i], &CoreB[0], 32 * 32 * sizeof(float));
    std::memcpy(&C[32 * 32 * i], &CoreC[0], 32 * 32 * sizeof(float));
  }

  float *A_dev = nullptr;
  float *B_dev = nullptr;
  float *C1_dev = nullptr;
  float *C2_dev = nullptr;

  std::cout << "Allocating device memory" << std::endl;
  cudaMalloc((void **)&B_dev, sizeof(float) * 32 * 32 * 156728);
  CHECK_ERR;
  cudaMalloc((void **)&A_dev, sizeof(float) * 32 * 32 * 156728);
  CHECK_ERR;
  cudaMalloc((void **)&C1_dev, sizeof(float) * 32 * 32 * 156728);
  CHECK_ERR;
  cudaMalloc((void **)&C2_dev, sizeof(float) * 32 * 32 * 156728);
  CHECK_ERR;

  std::cout << "Copying buffers to device" << std::endl;
  cudaMemcpy((void *)B_dev, (void *)B, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;

  // Dense x Dense Matrix Mult
  std::cout << "Calling Dense x Dense kernel" << std::endl;
  // First one the be discarded
  A_B_DenseXDense(A_dev, 0, B_dev, 0, C1_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  float elapsedTime = 0.0;
  cudaEvent_t startDD, stopDD;
  cudaEventCreate(&startDD);
  CHECK_ERR;
  cudaEventCreate(&stopDD);
  CHECK_ERR;
  cudaEventRecord(startDD);
  CHECK_ERR;
  A_B_DenseXDense(A_dev, 0, B_dev, 0, C1_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopDD);
  CHECK_ERR;
  cudaEventSynchronize(stopDD);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDD, stopDD);
  CHECK_ERR;
  std::cout << "Dense x Dense kernel took " << elapsedTime << " ms" << std::endl;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy(R1, C1_dev, sizeof(float) * 32 * 32 * 156728, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaEventDestroy(startDD);
  cudaEventDestroy(stopDD);

  // Sparse x Dense Matrix Mult Ellpack
  std::cout << "Calling Densse x Dense CO" << std::endl;
  // First one the be discarded
  A_B_DenseXDense_CO(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  elapsedTime = 0.0;
  cudaEvent_t startDS, stopDS;
  cudaEventCreate(&startDS);
  CHECK_ERR;
  cudaEventCreate(&stopDS);
  CHECK_ERR;
  cudaEventRecord(startDS);
  CHECK_ERR;
  A_B_DenseXDense_CO(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopDS);
  CHECK_ERR;
  cudaEventSynchronize(stopDS);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
  CHECK_ERR;
  std::cout << "Dense x Dense kernel CO took " << elapsedTime << " ms" << std::endl;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy(R2, C2_dev, sizeof(float) * 32 * 32 * 156728, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaEventDestroy(startDS);
  cudaEventDestroy(stopDS);

  bool EllpackDiff = false;
  for (int i = 0; i < 32 * 32 * 156728; i++)
  {
    if (R1[i] >= R2[i] + 0.001 || R1[i] <= R2[i] - 0.001)
    {
      // throw std::runtime_error(" Dense x  Dense and  Dense x  Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
      // std::cout << "RESULTS DONT MATCH" << std::endl;
      // return 0;
      EllpackDiff = true;
      break;
    }
    else
    {
      R2[i] = 0.0;
    }
  }
  if (!EllpackDiff)
  {
    std::cout << "Results Match! Ellpac" << std::endl;
  }
  else
  {
    std::cout << "Results DO NOT Match! Ellpac" << std::endl;
  }

  std::cout << "Calling Densse x Sparse kernel Distribute Columns of B" << std::endl;
  // Wipe the values of C2
  // First one the be discarded
  A_full_B_SparseXDense_Distribute_Columns_of_B(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  elapsedTime = 0.0;
  cudaEventCreate(&startDS);
  CHECK_ERR;
  cudaEventCreate(&stopDS);
  CHECK_ERR;
  cudaEventRecord(startDS);
  CHECK_ERR;
  A_full_B_SparseXDense_Distribute_Columns_of_B(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopDS);
  CHECK_ERR;
  cudaEventSynchronize(stopDS);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
  CHECK_ERR;
  std::cout << "Sparse x Dense kernel Distributed Columns of B took " << elapsedTime << " ms" << std::endl;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy(R2, C2_dev, sizeof(float) * 32 * 32 * 156728, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaEventDestroy(startDS);
  cudaEventDestroy(stopDS);

  bool DistributeColumnsOfBdiff = false;
  for (int i = 0; i < 32 * 32 * 156728; i++)
  {
    if (R1[i] >= R2[i] + 0.001 || R1[i] <= R2[i] - 0.001)
    {
      // throw std::runtime_error(" Dense x  Dense and  Dense x  Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
      // std::cout << "RESULTS DONT MATCH" << std::endl;
      // return 0;
      DistributeColumnsOfBdiff = true;
      break;
    }
    else
    {
      R2[i] = 0.0;
    }
  }
  if (!DistributeColumnsOfBdiff)
  {
    std::cout << "Results Match! Distribute Columns Of B" << std::endl;
  }
  else
  {
    std::cout << "Results DO NOT Match! Distribute Columns Of B" << std::endl;
  }

  std::cout << "Calling Densse x Sparse kernel Distribute Rows of B" << std::endl;
  // First one the be discarded
  A_full_B_SparseXDense_Distribute_Rows_of_B(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  elapsedTime = 0.0;
  cudaEventCreate(&startDS);
  CHECK_ERR;
  cudaEventCreate(&stopDS);
  CHECK_ERR;
  cudaEventRecord(startDS);
  CHECK_ERR;
  A_full_B_SparseXDense_Distribute_Rows_of_B(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopDS);
  CHECK_ERR;
  cudaEventSynchronize(stopDS);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
  CHECK_ERR;
  std::cout << "Sparse x Dense kernel Distributed Rows of B took " << elapsedTime << " ms" << std::endl;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy(R2, C2_dev, sizeof(float) * 32 * 32 * 156728, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaEventDestroy(startDS);
  cudaEventDestroy(stopDS);

  bool DistributeRowsOfBdiff = false;
  int problemAt = -1;
  for (int i = 0; i < 32 * 32 * 156728; i++)
  {
    if (R1[i] >= R2[i] + 0.001 || R1[i] <= R2[i] - 0.001)
    {
      // throw std::runtime_error(" Dense x  Dense and  Dense x  Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
      // std::cout << "RESULTS DONT MATCH" << std::endl;
      // return 0;
      DistributeRowsOfBdiff = true;
      problemAt = i;
      break;
    }
    else
    {
      R2[i] = 0.0;
    }
  }
  if (!DistributeRowsOfBdiff)
  {
    std::cout << "Results Match! Distribute Rows Of B" << std::endl;
  }
  else
  {
    std::cout << "Results DO NOT Match! Distribute Rows Of B first difference at offset " << problemAt << std::endl;
  }

  std::cout << "Calling Densse x Sparse kernel Distribute Columns of B double shared memory" << std::endl;
  // First one the be discarded
  A_full_B_SparseXDense_Distribute_Cols_of_B_double_load(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * 32 * 32 * 156728, cudaMemcpyHostToDevice);
  CHECK_ERR;
  elapsedTime = 0.0;
  cudaEventCreate(&startDS);
  CHECK_ERR;
  cudaEventCreate(&stopDS);
  CHECK_ERR;
  cudaEventRecord(startDS);
  CHECK_ERR;
  A_full_B_SparseXDense_Distribute_Cols_of_B_double_load(A_dev, 0, B_dev, 0, C2_dev, 0, 156728, nullptr, nullptr);
  CHECK_ERR;
  cudaEventRecord(stopDS);
  CHECK_ERR;
  cudaEventSynchronize(stopDS);
  CHECK_ERR;
  cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
  CHECK_ERR;
  std::cout << "Sparse x Dense kernel Distributed Columns of B double shared memory took " << elapsedTime << " ms" << std::endl;
  cudaDeviceSynchronize();
  CHECK_ERR;
  cudaMemcpy(R2, C2_dev, sizeof(float) * 32 * 32 * 156728, cudaMemcpyDeviceToHost);
  CHECK_ERR;
  cudaEventDestroy(startDS);
  cudaEventDestroy(stopDS);

  DistributeRowsOfBdiff = false;
  problemAt = -1;
  for (int i = 0; i < 32 * 32 * 156728; i++)
  {
    if (R1[i] >= R2[i] + 0.001 || R1[i] <= R2[i] - 0.001)
    {
      DistributeRowsOfBdiff = true;
      problemAt = i;
      break;
    }
    else
    {
      R2[i] = 0.0;
    }
  }
  if (!DistributeRowsOfBdiff)
  {
    std::cout << "Results Match! Distribute Columns of B double shared memory" << std::endl;
  }
  else
  {
    std::cout << "Results DO NOT Match! Distribute Columns of B double shared memory first difference at offset " << problemAt << std::endl;
  }

  std::cout << "Freeing device memory" << std::endl;
  cudaFree(B_dev);
  cudaFree(A_dev);
  cudaFree(C1_dev);
  cudaFree(C2_dev);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] R1;
  delete[] R2;
}