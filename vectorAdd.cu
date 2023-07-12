#include <iostream>
#include <cuda_runtime.h>

#define N 200000

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
    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * sizeof(float));
    cudaMalloc((void**)&dev_c, N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(dev_a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, vectorAdd, 0, N);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c);

    // Stop timer and calculate execution duration
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result from device to host
    cudaMemcpy(host_c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print execution duration
    std::cout << "Kernel execution duration: " << milliseconds << " ms" << std::endl;

    int numFloatingPointOps = N;
    int numBytesAccessed = 3 * N * sizeof(float);
    float opsPerByte = static_cast<float>(numFloatingPointOps) / static_cast<float>(numBytesAccessed);

    std::cout << "Floating-point operations per byte: " << opsPerByte << std::endl;

    float executionTimeSeconds = milliseconds / 1e3;
    float numGFLOPs = static_cast<float>(numFloatingPointOps) / 1e9;
    float GFLOPs = numGFLOPs / executionTimeSeconds;

    std::cout << "GFLOP/s: " << GFLOPs << std::endl;

    // Cleanup
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}