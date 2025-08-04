#pragma once

#ifdef __NVCC__

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

    
#define BLOCK_SIZE 256

inline dim3 calculate_grid_block(size_t size, int block_size = BLOCK_SIZE) {
    const int grid_size = (size + block_size - 1) / block_size;
    return dim3(grid_size, 1, 1);
}

extern __shared__ unsigned char shared_data[];
    
#endif 