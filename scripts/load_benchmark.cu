#include <cuda_runtime.h>
#include <iostream>

constexpr size_t NUM_ELEMENTS = 100000;
constexpr size_t N = 16;

__global__ __launch_bounds__(N) void load_tensor_slices(float *__restrict__ memory, size_t leadingDim, size_t N, float* __restrict__ accum)
{
    __shared__ __align__(8) float sharedMem[16 * 16 * 16];
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            sharedMem[threadIdx.x + j * N] = memory[threadIdx.x + j * leadingDim];
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            sharedMem[threadIdx.x + j * N] += i * j;
        }
    }
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            accum[threadIdx.x + j * N] = sharedMem[threadIdx.x + j * N];
        }
    }
}

int main()
{
    float *host_tensors = new float[N * N * N * NUM_ELEMENTS];
    float *device_tensors = nullptr;
    float *device_accum = nullptr;

    cudaMalloc((void **)&device_tensors, sizeof(float) * N * N * N * NUM_ELEMENTS);
    cudaMalloc((void **)&device_accum, sizeof(float) * N * N * N);
    cudaMemcpy(device_tensors, host_tensors, sizeof(float) * N * N * N * NUM_ELEMENTS, cudaMemcpyHostToDevice);

    constexpr size_t load_tensor_count = NUM_ELEMENTS / 4096;
    size_t leadingDimensions[] = {32, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    float elapsedTimes[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (size_t k = 0; k < 150; k++)
    {
        for (size_t i = 0; i < 9; i++)
        {
            size_t leadingDim = leadingDimensions[i];
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // Record the start event
            cudaEventRecord(start);
            dim3 block(N, 1, 1);
            dim3 grid(load_tensor_count, 1, 1);
            load_tensor_slices<<<grid, block, 0>>>(device_tensors, leadingDim, N, device_accum);
            cudaDeviceSynchronize();
            // Record the stop event
            cudaEventRecord(stop);

            // Synchronize to make sure the stop event is recorded
            cudaEventSynchronize(stop);

            // Calculate the elapsed time
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            elapsedTimes[i] += milliseconds;
        }
    }

    for (size_t i = 1; i < 9; i++)
    {
        std::cout << "Elapsed Time for case " << i << " on avg.: " << elapsedTimes[i] / 150 << std::endl;
    }

    cudaFree(device_tensors);
    delete[] host_tensors;
}