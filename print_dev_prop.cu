#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) { cores = mp * 128; assert(cores == 2048); }
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int main()
{
    cudaDeviceProp deviceProp;
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device Name: " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << "MB" << std::endl;
        std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max Blocks per Multiprocessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "Max Block Dimension (x, y, z): (" << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max Grid Dimension (x, y, z): (" << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << "MHz" << std::endl;
        std::cout << "Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "Peak Memory Bandwidth: " << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6
                  << "GB/s" << std::endl;
        // Calculate peak floating-point performance
        int coreCount = getSPcores(deviceProp);
        std::cout << "CUDA Cores: " << coreCount << std::endl;
        //int cudaCores = deviceProp.multiProcessorCount * coreCount * deviceProp.maxBlocksPerMultiProcessor;
        
        float clockSpeedHz = deviceProp.clockRate * 1e3;
        std::cout << "Clock Rate: " << clockSpeedHz / 1e9 << " Ghz" << std::endl;
        float flopsPerCUDACore = 2;  // Assuming FMA instructions (multiply + add)
        float peakPerformance = coreCount * clockSpeedHz * flopsPerCUDACore;
        std::cout << "Peak Floating-Point Performance: " << peakPerformance / 1e9 << " GFLOP/s" << std::endl;
        std::cout << "------------------------------------" << std::endl;
    }

    return 0;
}
