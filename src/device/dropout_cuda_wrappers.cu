#ifndef DROPOUT_CUDA_WRAPPERS_CU
#define DROPOUT_CUDA_WRAPPERS_CU

#include "../../inc/device/dropout_cuda_wrappers.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <ctime>

__global__ void setup_curand_kernel_dropout(curandState *state, unsigned long seed, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) curand_init(seed, idx, 0, &state[idx]);
}

__global__ void dropout_mask_kernel_impl(curandState *state, bool* mask, float dropout_prob, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        curandState localState = state[idx];
        float random_val = curand_uniform(&localState);
        mask[idx] = random_val > dropout_prob;
        state[idx] = localState;
    }
}

template<typename T>
__global__ void dropout_forward_kernel_impl(const T* input, const bool* mask, T* output, T scale, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = mask[idx] ? input[idx] * scale : static_cast<T>(0);
}

template<typename T>
__global__ void dropout_backward_kernel_impl(const T* grad_output, const bool* mask, T* grad_input, T scale, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) grad_input[idx] = mask[idx] ? grad_output[idx] * scale : static_cast<T>(0);
}

namespace cuda_dropout_ops {

template<typename T>
class DropoutCudaState {
private:
    curandState* d_state;
    size_t size;
    bool initialized;

public:
    DropoutCudaState() : d_state(nullptr), size(0), initialized(false) {}
    
    ~DropoutCudaState() {
        if(d_state) {
            cudaFree(d_state);
        }
    }
    
    void initialize(size_t tensor_size) {
        if(!initialized || size != tensor_size) {
            if(d_state) {
                cudaFree(d_state);
            }
            
            size = tensor_size;
            cudaMalloc(&d_state, size * sizeof(curandState));
            
            const dim3 grid = calculate_grid_block(size);
            
            unsigned long seed = static_cast<unsigned long>(time(NULL));
            setup_curand_kernel_dropout<<<grid, BLOCK_SIZE>>>(d_state, seed, size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            initialized = true;
        }
    }
    
    curandState* get_state() { return d_state; }
};

template<typename T>
thread_local DropoutCudaState<T> global_dropout_state;

template<typename T>
void launch_dropout_mask(bool* mask, T dropout_rate, size_t size) {
    global_dropout_state<T>.initialize(size);
    
    const dim3 grid = calculate_grid_block(size);
    
    float dropout_prob = static_cast<float>(dropout_rate);
    dropout_mask_kernel_impl<<<grid, BLOCK_SIZE>>>(global_dropout_state<T>.get_state(), mask, dropout_prob, size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_dropout_forward(const T* input, const bool* mask, T* output, T scale, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    dropout_forward_kernel_impl<<<grid, BLOCK_SIZE>>>(input, mask, output, scale, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_dropout_backward(const T* grad_output, const bool* mask, T* grad_input, T scale, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    dropout_backward_kernel_impl<<<grid, BLOCK_SIZE>>>(grad_output, mask, grad_input, scale, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit template instantiations
template void launch_dropout_mask<float>(bool*, float, size_t);
template void launch_dropout_mask<double>(bool*, double, size_t);

template void launch_dropout_forward<float>(const float*, const bool*, float*, float, size_t);
template void launch_dropout_forward<double>(const double*, const bool*, double*, double, size_t);

template void launch_dropout_backward<float>(const float*, const bool*, float*, float, size_t);
template void launch_dropout_backward<double>(const double*, const bool*, double*, double, size_t);

}

#endif
