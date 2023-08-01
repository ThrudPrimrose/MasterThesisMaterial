// Example Tensor Contraction
// Number 1, tensor Multiplication as a Tensor:
// C[ij] = A[ik] * B[kj]
// Number 2, 3D to 3D Tensors
// C[nko] = A[mko] * B[nmo]

#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <cutensor.h>

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

bool compareMatrices(const float *tensorA, const float *tensorB, size_t numElements, float tolerance)
{
    for (int i = 0; i < numElements; i++)
    {
        float diff = std::fabs(tensorA[i] - tensorB[i]);
        if (diff > tolerance)
        {
            return false;
        }
    }

    return true;
}

void matrixMultiplyCPU(const float *matrixA, const float *matrixB, float *matrixC, int numRows, int numCols, int sharedDim)
{
    for (int col = 0; col < numCols; col++)
    {
        for (int row = 0; row < numRows; row++)
        {
            float sum = 0.0f;
            for (int k = 0; k < sharedDim; k++)
            {
                sum += matrixA[k * numRows + row] * matrixB[k + numRows * col];
            }
            matrixC[col * numRows + row] = sum;
        }
    }
}

// Print a tensor
void printTensor(const float *tensor, int numRows, int numCols, int numZ)
{
    std::cout << std::setprecision(4);
    for (int z = 0; z < numZ; z++)
    {
        for (int row = 0; row < numRows; row++)
        {
            for (int col = 0; col < numCols; col++)
            {
                std::cout << tensor[col * numRows + row] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

__global__ void 
__launch_bounds__(96)
 kernel_sgemm_NT_NT_m8_n8_k8_lda8_ldb8_ldc8_alpha_1_beta_0_ppp_84fcd7e(const float * A, int A_extraOffset, const float * B, int B_extraOffset, float * C, int C_extraOffset, unsigned numElements, unsigned* flags) {
  unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
  if (batchID < numElements) {
    bool isFlagsProvided = (flags != nullptr);
    bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
    if (allowed) {
      const float * const __restrict__ glb_A = &A[0 /* + MatrixOffset + */ + A_extraOffset];
      const float * const __restrict__ glb_B = &B[0 /* + MatrixOffset + */ + B_extraOffset];
      float * const __restrict__ glb_C = &C[0 /* + MatrixOffset + */ + C_extraOffset];
      float reg0[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      __shared__  __align__(8) float totalShrMem[192];
      float * localShrMem0 = &totalShrMem[64 * threadIdx.y];

      float* shrRegion0 = &localShrMem0[0];
      // using ExtendedPatchLoader
      {
        shrRegion0[threadIdx.x + 0] = glb_B[threadIdx.x + 0];
        shrRegion0[threadIdx.x + 32] = glb_B[threadIdx.x + 32];
      }
      __syncwarp();
      if (threadIdx.x < 8) {
        float value;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
          value = glb_A[threadIdx.x + k * 8];

          #pragma unroll
          for (int n = 0; n < 8; ++n) {
            reg0[n] += value * shrRegion0[k + 8 * n];
          }
        }
      }
      if (threadIdx.x < 8) {
        // store.py writing things
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          glb_C[threadIdx.x + 8 * n] = reg0[n]; //u
        }
      }
    }
  }
}
void sgemm_NT_NT_m8_n8_k8_lda8_ldb8_ldc8_alpha_1_beta_0_ppp_84fcd7e(const float *A, int A_extraOffset, const float *B, int B_extraOffset, float *C, int C_extraOffset, unsigned numElements, unsigned* flags, void* streamPtr) {
  dim3 block(32, 3, 1);
  dim3 grid((numElements + 3 - 1) / 3, 1, 1);
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_sgemm_NT_NT_m8_n8_k8_lda8_ldb8_ldc8_alpha_1_beta_0_ppp_84fcd7e<<<grid,block,0,stream>>>(A, A_extraOffset, B, B_extraOffset, C, C_extraOffset, numElements, flags);
  CHECK_ERR;
}



int main()
{
    const int64_t dims[3] = {8, 8, 8};
    const int64_t numElements = dims[0] * dims[1] * dims[2];
    const int64_t tensorSize = numElements * sizeof(float);
    const float tolerance = 1e-6; // Tolerance for floating-point comparison
    const int64_t numTensors = 1000;

    // Initialize the column-major matrices A, B, and C
    float **tensorA = new float*[numTensors];
    float **tensorB = new float*[numTensors];
    float **tensorC_CPU = new float*[numTensors];
    float **tensorC_cuTensor = new float*[numTensors];
    float **tensorC_LoG = new float*[numTensors];

    for (int i = 0; i<numTensors; i++){
        tensorA[i] = new float[numElements];
        tensorB[i] = new float[numElements];
        tensorC_CPU[i] = new float[numElements];
        tensorC_cuTensor[i] = new float[numElements];
        tensorC_LoG[i] = new float[numElements];
    }

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize matrices A and B with random values
    for (int t = 0; t < numTensors; t++){
        for (int i = 0; i < numElements; i++)
        {
            tensorA[t][i] = dist(gen);
            tensorB[t][i] = dist(gen);
            tensorC_CPU[t][i] = 0.f;
            tensorC_cuTensor[t][i] = 0.f;
            tensorC_LoG[t][i] = 0.f;
        }
    }


    // Mat mul CPU
    for (int t = 0; t < numTensors; t++)
    {
        for (size_t z = 0; z < dims[2]; z++)
        {
            matrixMultiplyCPU(tensorA[t] + z * dims[0] * dims[1],
                              tensorB[t] + z * dims[0] * dims[1],
                              tensorC_CPU[t] + z * dims[0] * dims[1],
                              dims[0], dims[1], dims[0]);
        }
    }

    // Mat mul with my general Gemm Implementation
    {
        float **tmpDevicetensorA = new float*[numTensors];
        float **tmpDevicetensorB = new float*[numTensors];
        float **tmpDevicetensorC = new float*[numTensors];
        float **devicetensorA;
        float **devicetensorB;
        float **devicetensorC;

        //cudaMalloc((void **)&devicetensorA, numTensors * sizeof(float*));
        //CHECK_ERR;
        //cudaMalloc((void **)&devicetensorB, numTensors * sizeof(float*));
        //CHECK_ERR;
        //cudaMalloc((void **)&devicetensorC, numTensors * sizeof(float*));
        //CHECK_ERR;

        for (int t = 0; t < numTensors; t++){
            cudaMalloc((void **)&tmpDevicetensorA[t], tensorSize * sizeof(float));
            CHECK_ERR;
            cudaMalloc((void **)&tmpDevicetensorB[t], tensorSize * sizeof(float));
            CHECK_ERR;
            cudaMalloc((void **)&tmpDevicetensorC[t], tensorSize * sizeof(float));
            CHECK_ERR;
            cudaMemcpy(tmpDevicetensorA[t], tensorA[t], tensorSize * sizeof(float), cudaMemcpyHostToDevice);
            CHECK_ERR;
            cudaMemcpy(tmpDevicetensorB[t], tensorB[t], tensorSize * sizeof(float), cudaMemcpyHostToDevice);
            CHECK_ERR;
        }

        cudaMalloc((void **)&devicetensorA, numTensors * sizeof(float*));
        CHECK_ERR;
        cudaMalloc((void **)&devicetensorB, numTensors * sizeof(float*));
        CHECK_ERR;
        cudaMalloc((void **)&devicetensorC, numTensors * sizeof(float*));
        CHECK_ERR;
        cudaMemcpy(devicetensorA, tmpDevicetensorA, numTensors * sizeof(float*), cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(devicetensorB, tmpDevicetensorB, numTensors * sizeof(float*), cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(devicetensorC, tmpDevicetensorC, numTensors * sizeof(float*), cudaMemcpyHostToDevice);
        CHECK_ERR;

        // Perform tensor multiplication C = A * B using cuBLAS
        //for (int t = 0; t < numTensors; ++t) {
            for (int _o = 0; _o < 8; ++_o) {
                float const* _A = tmpDevicetensorA[t] + 64*_o;
                float const* _B =  tmpDevicetensorB[t] + 64*_o;
                float * _C =  tmpDevicetensorC[t] + 64*_o;
                sgemm_NT_NT_m8_n8_k8_lda8_ldb8_ldc8_alpha_1_beta_0_ppp_84fcd7e(const_cast<const float *>(_A), 0, const_cast<const float *>(_B), 0, _C, 0, numElements, nullptr, nullptr);
            }
        //}
        cudaDeviceSynchronize();

        for (int t = 0; t < numTensors; t++){
            // Copy the result tensor C from the GPU to the CPU
            cudaMemcpy(tensorC_LoG[t], tmpDevicetensorC[t], tensorSize * sizeof(float), cudaMemcpyDeviceToHost);
            CHECK_ERR;
            // Compare results with CPU tensor multiplication
            bool resultsMatch = compareMatrices(tensorC_CPU[t], tensorC_LoG[t], numElements, tolerance);
            if (resultsMatch)
            {
                //std::cout << "Results match! (CPU-cuBLAS)" << std::endl;
            }
            else
            {
                std::cout << "Results do not match! (CPU <-> LoG) at " << t << std::endl;
            }
        }


        // Clean up resources
        cudaFree(devicetensorA);
        CHECK_ERR;
        cudaFree(devicetensorB);
        CHECK_ERR;
        cudaFree(devicetensorC);
        CHECK_ERR;
    }

    // tensor multiplication with cuTensor
    {
        // cuTensor initialization
        cutensorHandle_t handle;
        cutensorInit(&handle);
        CHECK_ERR;

        // Create vector of modes
        std::vector<int> modeA{'i', 'k', 'n'};
        std::vector<int> modeB{'k', 'j', 'n'};
        std::vector<int> modeC{'i', 'j', 'n'};
        int nmodeA = modeA.size();
        int nmodeB = modeB.size();
        int nmodeC = modeC.size();

        // Tensor descriptors
        cutensorTensorDescriptor_t descA, descB, descC;
        const int64_t *extentA = new int64_t[3]{dims[0], dims[1], dims[2]};
        const int64_t *extentB = new int64_t[3]{dims[0], dims[1], dims[2]};
        const int64_t *extentC = new int64_t[3]{dims[0], dims[1], dims[2]};

        // size_t elementsA = numRows * sharedDim;
        // size_t elementsB = sharedDim * numCols;
        // size_t elementsC = numRows * numCols;

        float *devicetensorA;
        float *devicetensorB;
        float *devicetensorC;

        cudaMalloc((void **)&devicetensorA, tensorSize);
        CHECK_ERR;
        cudaMalloc((void **)&devicetensorB, tensorSize);
        CHECK_ERR;
        cudaMalloc((void **)&devicetensorC, tensorSize);
        CHECK_ERR;

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Copy matrices A and B from the CPU to the GPU
        cudaMemcpy(devicetensorA, tensorA[0], tensorSize, cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(devicetensorB, tensorB[0], tensorSize, cudaMemcpyHostToDevice);
        CHECK_ERR;

        cutensorInitTensorDescriptor(&handle, &descA, 3, extentA, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
        CHECK_ERR;
        cutensorInitTensorDescriptor(&handle, &descB, 3, extentB, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
        CHECK_ERR;
        cutensorInitTensorDescriptor(&handle, &descC, 3, extentC, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
        CHECK_ERR;

        uint32_t alignmentRequirementA;
        uint32_t alignmentRequirementB;
        uint32_t alignmentRequirementC;
        cutensorGetAlignmentRequirement(&handle,
                                        devicetensorA,
                                        &descA,
                                        &alignmentRequirementA);
        CHECK_ERR;
        cutensorGetAlignmentRequirement(&handle,
                                        devicetensorB,
                                        &descB,
                                        &alignmentRequirementB);
        CHECK_ERR;
        cutensorGetAlignmentRequirement(&handle,
                                        devicetensorC,
                                        &descC,
                                        &alignmentRequirementC);
        CHECK_ERR;

        // cuTensor contraction
        cutensorContractionDescriptor_t desc;
        cutensorInitContractionDescriptor(&handle,
                                          &desc,
                                          &descA, modeA.data(), alignmentRequirementA,
                                          &descB, modeB.data(), alignmentRequirementB,
                                          &descC, modeC.data(), alignmentRequirementC,
                                          &descC, modeC.data(), alignmentRequirementC,
                                          CUTENSOR_COMPUTE_32F);
        CHECK_ERR;

        cutensorContractionFind_t find;
        cutensorInitContractionFind(
            &handle, &find,
            CUTENSOR_ALGO_DEFAULT);
        CHECK_ERR;

        size_t worksize = 0;
        cutensorContractionGetWorkspaceSize(&handle,
                                            &desc,
                                            &find,
                                            CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);
        CHECK_ERR;
        // Allocate workspace
        void *work = nullptr;
        if (worksize > 0)
        {
            if (cudaSuccess != cudaMalloc(&work, worksize)) // This is optional!
            {
                work = nullptr;
                worksize = 0;
            }
        }

        cutensorContractionPlan_t plan;
        cutensorInitContractionPlan(&handle,
                                    &plan,
                                    &desc,
                                    &find,
                                    worksize);
        CHECK_ERR;

        cutensorStatus_t err;

        // Execute the tensor contraction
        err = cutensorContraction(&handle,
                                  &plan,
                                  (void *)&alpha, devicetensorA,
                                  devicetensorB,
                                  (void *)&beta, devicetensorC,
                                  devicetensorC,
                                  work, worksize, 0 /* stream */);
        CHECK_ERR;

        cudaDeviceSynchronize();
        CHECK_ERR;

        cudaMemcpy(tensorC_cuTensor, devicetensorC, tensorSize, cudaMemcpyDeviceToHost);
        CHECK_ERR;

        cudaFree(work);
        CHECK_ERR;
        cudaFree(devicetensorA);
        CHECK_ERR;
        cudaFree(devicetensorB);
        CHECK_ERR;
        cudaFree(devicetensorC);
        CHECK_ERR;

        /*
        bool resultsMatch = compareMatrices(tensorC_cuTensor[0], tensorC_CPU[0], numElements, tolerance);

        if (resultsMatch)
        {
            std::cout << "Results match! (CPU <-> cuTensor)" << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (CPU <-> cuTensor)" << std::endl;
        }
        */
    }

    std::cout << "tensor C (CPU Result):" << std::endl;
    printTensor(tensorC_CPU[0], dims[0], dims[1], dims[2]);
    std::cout << std::endl;
    std::cout << "tensor C (GPU Result cuTensor):" << std::endl;
    printTensor(tensorC_cuTensor[0], dims[0], dims[1], dims[2]);
    std::cout << std::endl;
    std::cout << "tensor C (GPU Result LoG):" << std::endl;
    printTensor(tensorC_LoG[0], dims[0], dims[1], dims[2]);
    std::cout << std::endl;

    delete[] tensorA;
    delete[] tensorB;
    delete[] tensorC_CPU;
    delete[] tensorC_cuTensor;
}