#include <cuda_runtime.h>
#include <iostream>

const int THREADS_PER_BLOCK = 1024;



#if !defined(UNROLL_COUNT)
#define UNROLL_COUNT 1
#endif

#if !defined(ITERARTIONS)
#define ITERARTIONS 1000000
#endif

constexpr size_t unroll_count = UNROLL_COUNT;
constexpr size_t iter_count = ITERARTIONS;

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

__global__ void incrementKernel(float *A, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        #pragma unroll unroll_count
        for (size_t i = 0; i < iter_count; i++) {
            A[idx] += 1.0f;
        }
    }
}

int main() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem); CHECK_ERR;

    std::cout << "Free Memory (bytes): " << free_mem << std::endl;
    std::cout << "Total Memory (bytes): " << total_mem << std::endl;


    size_t N = static_cast<size_t>((0.9 * free_mem)) / sizeof(float);
    float *h_A = new float[N];
    float *d_A;

    // Initialize the host array
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate device memory and copy data from host to device
    cudaMalloc((void**)&d_A, N * sizeof(float)); CHECK_ERR;
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;

    // Launch the kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Define CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start); CHECK_ERR;
    cudaEventCreate(&stop); CHECK_ERR;
    // Record start event
    cudaEventRecord(start); CHECK_ERR;
    incrementKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, N); CHECK_ERR;
    // Record stop event
    cudaEventRecord(stop); CHECK_ERR;
    // Wait for the stop event to complete
    cudaEventSynchronize(stop); CHECK_ERR;

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); CHECK_ERR;

    // Copy back the result to the host
    cudaMemcpy(h_A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost); CHECK_ERR;

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    // Cleanup
    delete[] h_A;
    cudaFree(d_A); CHECK_ERR;
    // Cleanup CUDA events
    cudaEventDestroy(start); CHECK_ERR;
    cudaEventDestroy(stop); CHECK_ERR;

    return 0;
}