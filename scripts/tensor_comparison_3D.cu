// Example Tensor Contraction
// Number 1, Matrix Multiplication as a Tensor:
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

// Print a matrix
void printTensor(const float *matrix, int numRows, int numCols, int numZ)
{
    std::cout << std::setprecision(4);
    for (int z = 0; z < numZ; z++)
    {
        for (int row = 0; row < numRows; row++)
        {
            for (int col = 0; col < numCols; col++)
            {
                std::cout << matrix[col * numRows + row] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

__global__ void
    __launch_bounds__(32)
        gemm(const float *A, const int offsetBetweenElementsAx, const int offsetBetweenElementsAy,
             const float *B, const int offsetBetweenElementsBx, const int offsetBetweenElementsBy,
             float *C, const int offsetBetweenElementsCx, const int offsetBetweenElementsCy,
             unsigned numElements, unsigned *flags)
{
    unsigned batchID = (threadIdx.y + blockDim.y * blockIdx.x);
    if (batchID < numElements)
    {
        bool isFlagsProvided = (flags != nullptr);
        bool allowed = isFlagsProvided ? static_cast<bool>(flags[batchID]) : true;
        if (allowed)
        {
            const float *const __restrict__ glb_A = &A[batchID * 64 + 0];
            const float *const __restrict__ glb_B = &B[batchID * 64 + 0];
            float *const __restrict__ glb_C = &C[batchID * 64 + 0];
            float reg0[8] = {0.0f};
            __shared__ __align__(8) float totalShrMem[64];
            float *localShrMem0 = &totalShrMem[64 * threadIdx.y];

            float *shrRegion0 = &localShrMem0[0];
            /*
                assertions that offsets are either matches 1 or a combination of dimensions like dimA or dimB or dimC*dimB etc.
            */
            // using ExtendedPatchLoader
            if (threadIdx.x < 8)
            {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    shrRegion0[threadIdx.x + i * 8] = glb_B[(threadIdx.x * offsetBetweenElementsBy) + (i * offsetBetweenElementsBx)];
                }
            }
            __syncwarp();
            if (threadIdx.x < 8)
            {
                float value;

#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    value = glb_A[(threadIdx.x * offsetBetweenElementsAy) + (k * offsetBetweenElementsAx)];

#pragma unroll
                    for (int n = 0; n < 8; ++n)
                    {
                        reg0[n] += value * shrRegion0[k + 8 * n];
                    }
                }
            }
            if (threadIdx.x < 8)
            {
#pragma unroll
                for (int n = 0; n < 8; ++n)
                {
                    glb_C[(threadIdx.x * offsetBetweenElementsCy) + (n * offsetBetweenElementsCx)] = reg0[n];
                }
            }
        }
    }
}

void gemm_launcher(const float *A, const int offsetBetweenElementsAx, const int offsetBetweenElementsAy,
                   const float *B, const int offsetBetweenElementsBx, const int offsetBetweenElementsBy,
                   float *C, const int offsetBetweenElementsCx, const int offsetBetweenElementsCy,
                   const int64_t *dims, const int64_t iterDimOffset, const int64_t numDims,
                   unsigned numElements, unsigned *flags, void *streamPtr)
{
    dim3 block(32, 1, 1);
    dim3 grid((numElements + 1 - 1) / 1, 1, 1);
    // cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
    int offset = 1;
    for (int i = 0; i < iterDimOffset; i++)
    {
        offset *= dims[i];
    }
    for (int iter = 0; iter < dims[iterDimOffset]; iter++)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        CHECK_ERR;
        gemm<<<grid, block, 0, stream>>>(
            A + iter * offset, offsetBetweenElementsAx, offsetBetweenElementsAy,
            B + iter * offset, offsetBetweenElementsBx, offsetBetweenElementsBy,
            C + iter * offset, offsetBetweenElementsCx, offsetBetweenElementsCy,
            numElements, flags);
        CHECK_ERR;
    }
    CHECK_ERR;
}

int main()
{
    const int64_t dims[3] = {8, 8, 8};
    const int64_t numElements = dims[0] * dims[1] * dims[2];
    const int64_t matrixSize = numElements * sizeof(float);
    const float tolerance = 1e-6; // Tolerance for floating-point comparison

    // Initialize the column-major matrices A, B, and C
    float *matrixA = new float[numElements];
    float *matrixB = new float[numElements];
    float *matrixC_CPU = new float[numElements];
    float *matrixC_GPU = new float[numElements];
    float *matrixC_cuTensor = new float[numElements];
    float *matrixC_LoG = new float[numElements];

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize matrices A and B with random values
    for (int i = 0; i < numElements; i++)
    {
        matrixA[i] = dist(gen);
        matrixB[i] = dist(gen);
        matrixC_CPU[i] = 0.f;
        matrixC_GPU[i] = 0.f;
        matrixC_cuTensor[i] = 0.f;
        matrixC_LoG[i] = 0.f;
    }

    // Mat mul CPU
    {
        for (size_t z = 0; z < dims[2]; z++)
        {
            matrixMultiplyCPU(matrixA + z * dims[0] * dims[1],
                              matrixB + z * dims[0] * dims[1],
                              matrixC_CPU + z * dims[0] * dims[1],
                              dims[0], dims[1], dims[0]);
        }
    }

    // Mat mul GPU with cuBLAS
    {
        // Transpose matrices A, B, and C to row-major format
        float *deviceMatrixA;
        float *deviceMatrixB;
        float *deviceMatrixC;

        cudaMalloc((void **)&deviceMatrixA, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixB, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixC, matrixSize);
        CHECK_ERR;

        // Multiply matrices A, B, and C using cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Copy matrices A and B from the CPU to the GPU
        cudaMemcpy(deviceMatrixA, matrixA, matrixSize, cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(deviceMatrixB, matrixB, matrixSize, cudaMemcpyHostToDevice);
        CHECK_ERR;

        // Perform matrix multiplication C = A * B using cuBLAS
        for (size_t z = 0; z < dims[2]; z++)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dims[0], dims[1], dims[0],
                        &alpha, deviceMatrixA + z * dims[0] * dims[1], dims[0],
                        deviceMatrixB + z * dims[0] * dims[1], dims[0],
                        &beta, deviceMatrixC + z * dims[0] * dims[1], dims[0]);
        }
        CHECK_ERR;
        cudaDeviceSynchronize();
        CHECK_ERR;

        // Copy the result matrix C from the GPU to the CPU
        cudaMemcpy(matrixC_GPU, deviceMatrixC, matrixSize, cudaMemcpyDeviceToHost);
        CHECK_ERR;

        // Compare results with CPU matrix multiplication

        bool resultsMatch = compareMatrices(matrixC_CPU, matrixC_GPU, numElements, tolerance);

        if (resultsMatch)
        {
            std::cout << "Results match! (CPU-cuBLAS)" << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (CPU-cuBLAS)" << std::endl;
        }

        // Clean up resources
        cudaFree(deviceMatrixA);
        CHECK_ERR;
        cudaFree(deviceMatrixB);
        CHECK_ERR;
        cudaFree(deviceMatrixC);
        CHECK_ERR;

        cublasDestroy(handle);
    }

    // Mat mul with my general Gemm Implementation
    {
        // Transpose matrices A, B, and C to row-major format
        float *deviceMatrixA;
        float *deviceMatrixB;
        float *deviceMatrixC;

        cudaMalloc((void **)&deviceMatrixA, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixB, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixC, matrixSize);
        CHECK_ERR;

        // Multiply matrices A, B, and C using cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Copy matrices A and B from the CPU to the GPU
        cudaMemcpy(deviceMatrixA, matrixA, matrixSize, cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(deviceMatrixB, matrixB, matrixSize, cudaMemcpyHostToDevice);
        CHECK_ERR;

        // Perform matrix multiplication C = A * B using cuBLAS
        gemm_launcher(deviceMatrixA, dims[0], 1,
                      deviceMatrixB, dims[0], 1,
                      deviceMatrixC, dims[0], 1,
                      &dims[0], 2, 3, 1, nullptr, nullptr);
        CHECK_ERR;
        cudaDeviceSynchronize();
        CHECK_ERR;

        // Copy the result matrix C from the GPU to the CPU
        cudaMemcpy(matrixC_LoG, deviceMatrixC, matrixSize, cudaMemcpyDeviceToHost);
        CHECK_ERR;

        // Compare results with CPU matrix multiplication

        bool resultsMatch = compareMatrices(matrixC_CPU, matrixC_LoG, numElements, tolerance);

        if (resultsMatch)
        {
            std::cout << "Results match! (CPU-OffsetGemm)" << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (CPU-OffsetGemm)" << std::endl;
        }

        // Clean up resources
        cudaFree(deviceMatrixA);
        CHECK_ERR;
        cudaFree(deviceMatrixB);
        CHECK_ERR;
        cudaFree(deviceMatrixC);
        CHECK_ERR;

        cublasDestroy(handle);
    }

    // Matrix multiplication with cuTensor
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

        float *deviceMatrixA;
        float *deviceMatrixB;
        float *deviceMatrixC;

        cudaMalloc((void **)&deviceMatrixA, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixB, matrixSize);
        CHECK_ERR;
        cudaMalloc((void **)&deviceMatrixC, matrixSize);
        CHECK_ERR;

        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        // Copy matrices A and B from the CPU to the GPU
        cudaMemcpy(deviceMatrixA, matrixA, matrixSize, cudaMemcpyHostToDevice);
        CHECK_ERR;
        cudaMemcpy(deviceMatrixB, matrixB, matrixSize, cudaMemcpyHostToDevice);
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
                                        deviceMatrixA,
                                        &descA,
                                        &alignmentRequirementA);
        CHECK_ERR;
        cutensorGetAlignmentRequirement(&handle,
                                        deviceMatrixB,
                                        &descB,
                                        &alignmentRequirementB);
        CHECK_ERR;
        cutensorGetAlignmentRequirement(&handle,
                                        deviceMatrixC,
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
                                  (void *)&alpha, deviceMatrixA,
                                  deviceMatrixB,
                                  (void *)&beta, deviceMatrixC,
                                  deviceMatrixC,
                                  work, worksize, 0 /* stream */);
        CHECK_ERR;

        cudaDeviceSynchronize();
        CHECK_ERR;

        cudaMemcpy(matrixC_cuTensor, deviceMatrixC, matrixSize, cudaMemcpyDeviceToHost);
        CHECK_ERR;

        cudaFree(work);
        CHECK_ERR;
        cudaFree(deviceMatrixA);
        CHECK_ERR;
        cudaFree(deviceMatrixB);
        CHECK_ERR;
        cudaFree(deviceMatrixC);
        CHECK_ERR;

        bool resultsMatch = compareMatrices(matrixC_cuTensor, matrixC_GPU, numElements, tolerance);

        if (resultsMatch)
        {
            std::cout << "Results match! (CPU-cuTensor)" << std::endl;
        }
        else
        {
            std::cout << "Results do not match! (CPU-cuTensor)" << std::endl;
        }
    }

    std::cout << "Matrix C (CPU Result):" << std::endl;
    printTensor(matrixC_CPU, dims[0], dims[1], dims[2]);
    std::cout << std::endl;
    std::cout << "Matrix C (GPU Result cuBLAS):" << std::endl;
    printTensor(matrixC_GPU, dims[0], dims[1], dims[2]);
    std::cout << std::endl;
    std::cout << "Matrix C (GPU Result cuTensor):" << std::endl;
    printTensor(matrixC_cuTensor, dims[0], dims[1], dims[2]);
    std::cout << std::endl;
    std::cout << "Matrix C (GPU Result LoG):" << std::endl;
    printTensor(matrixC_LoG, dims[0], dims[1], dims[2]);
    std::cout << std::endl;

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC_CPU;
    delete[] matrixC_GPU;
    delete[] matrixC_cuTensor;
}