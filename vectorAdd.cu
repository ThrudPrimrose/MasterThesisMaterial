#include <iostream>
#include <cuda_runtime.h>

constexpr size_t N = static_cast<size_t>(1e9 / static_cast<float>(sizeof(float)));

#define CHECK_ERR checkErr(__FILE__,__LINE__)

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

__global__ void vectorAdd(float* a, float* b, float* c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    // Declare and initialize host vectors
    float* host_a = new float[N];
    float* host_b = new float[N];
    float* host_c = new float[N];
    for (int i = 0; i < N; ++i)
    {
        host_a[i] = i;
        host_b[i] = 2 * i;
    }

    // Declare and allocate device vectors
    float* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, N * sizeof(float)); CHECK_ERR;
    cudaMalloc((void**)&dev_b, N * sizeof(float)); CHECK_ERR;
    cudaMalloc((void**)&dev_c, N * sizeof(float)); CHECK_ERR;

    // Copy host vectors to device
    cudaMemcpy(dev_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy(dev_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    // Define kernel launch configuration
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, vectorAdd, 0, N); CHECK_ERR;
    // int blockSize = 256;
    gridSize = (N + blockSize - 1) / blockSize;

    // Fire first kernel and discard
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c); CHECK_ERR;
    cudaDeviceSynchronize();

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start); CHECK_ERR;
    cudaEventCreate(&stop); CHECK_ERR;
    cudaEventRecord(start); CHECK_ERR;

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c); CHECK_ERR;

    // Stop timer and calculate execution duration
    cudaEventRecord(stop); CHECK_ERR;
    cudaEventSynchronize(stop); CHECK_ERR;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); CHECK_ERR;

    // Copy result from device to host
    cudaMemcpy(host_c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost); CHECK_ERR;
    cudaDeviceSynchronize(); CHECK_ERR;

    for (int i = 0; i < N; ++i)
    {
        if (host_c[i] > 1.001f * (3.0f * static_cast<float>(i)) ||
            host_c[i] < 0.999f * (3.0f * static_cast<float>(i))){
            throw std::runtime_error("Results different from expected " + std::to_string(host_c[i]) + " != " + std::to_string(3.0f * static_cast<float>(i)));
        }
    }

    // Print execution duration
    std::cout << "Kernel execution duration: " << milliseconds << " ms" << std::endl;

    size_t numFloatingPointOps = N;
    size_t numBytesAccessed = 3 * N * sizeof(float);
    float opsPerByte = static_cast<float>(numFloatingPointOps) / static_cast<float>(numBytesAccessed);

    std::cout << "Floating-point operations per byte: " << opsPerByte << std::endl;

    float executionTimeSeconds = milliseconds / 1e3;
    float numGFLOPs = static_cast<float>(numFloatingPointOps) / 1e9;
    float GFLOPs = numGFLOPs / executionTimeSeconds;

    std::cout << "GFLOP/s: " << GFLOPs << std::endl;

    float peakMemoryBandwidthTheo = 176.032; // GB /s
    float peakGFLOPTheo  = 4329.47; // GFlop /s
    float peakGFLOPforIntensity = std::min(peakMemoryBandwidthTheo * opsPerByte, peakGFLOPTheo);

    float achievedPeak = (static_cast<float>(GFLOPs) / peakGFLOPforIntensity) * 100.0f;
    std::string strAchievedPeak(6, '\0');
    std::sprintf(&strAchievedPeak[0], "%.2f", achievedPeak);
    std::cout << "Percentage of Peak Performance: " << strAchievedPeak << "%" << std::endl;

    float GBPerSecond = (static_cast<float>(numBytesAccessed) * 1e-9) / executionTimeSeconds;
    std::cout << "GB per Second: " << GBPerSecond << std::endl;

    // Cleanup
    cudaFree(dev_a); CHECK_ERR;
    cudaFree(dev_b); CHECK_ERR;
    cudaFree(dev_c); CHECK_ERR;
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}
