#ifndef ACTIVATION_CUDA_KERNELS_CU
#define ACTIVATION_CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

template<typename T>
__global__ void relu_apply_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = fmaxf(static_cast<T>(0), input[idx]);
}

template<typename T>
__global__ void relu_derivative_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = (input[idx] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
}

template<typename T>
__global__ void sigmoid_apply_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = static_cast<T>(1) / (static_cast<T>(1) + expf(-input[idx]));
}

template<typename T>
__global__ void sigmoid_derivative_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = input[idx] * (static_cast<T>(1) - input[idx]);
}

template<typename T>
__global__ void tanh_apply_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = tanhf(input[idx]);
}

template<typename T>
__global__ void tanh_derivative_kernel(const T* input, T* output, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = static_cast<T>(1) - input[idx] * input[idx];
}

template<typename T>
__global__ void elu_apply_kernel(const T* input, T* output, T alpha, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = (input[idx] > static_cast<T>(0)) ? input[idx] : alpha * (expf(input[idx]) - static_cast<T>(1));
}

template<typename T>
__global__ void elu_derivative_kernel(const T* input, T* output, T alpha, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = (input[idx] > static_cast<T>(0)) ? static_cast<T>(1) : alpha * expf(input[idx]);
}

template<typename T>
__global__ void leaky_relu_apply_kernel(const T* input, T* output, T alpha, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = (input[idx] > static_cast<T>(0)) ? input[idx] : alpha * input[idx];
}

template<typename T>
__global__ void leaky_relu_derivative_kernel(const T* input, T* output, T alpha, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) output[idx] = (input[idx] > static_cast<T>(0)) ? static_cast<T>(1) : alpha;
}

template<typename T>
__global__ void softmax_apply_kernel(const T* input, T* output, int batch_size, int num_classes, size_t size) {
    const int batch_idx = blockIdx.x;
    const int class_idx = threadIdx.x;
    
    if(batch_idx < batch_size && class_idx < num_classes) {
        T max_val = input[batch_idx * num_classes];
        for(int i = 1; i < num_classes; ++i) {
            max_val = fmaxf(max_val, input[batch_idx * num_classes + i]);
        }
        
        __shared__ T exp_values[1024]; 
        exp_values[class_idx] = expf(input[batch_idx * num_classes + class_idx] - max_val);
        __syncthreads();
        
        for(int stride = num_classes / 2; stride > 0; stride >>= 1) {
            if(class_idx < stride && class_idx + stride < num_classes) {
                exp_values[class_idx] += exp_values[class_idx + stride];
            }
            __syncthreads();
        }
        
        if(class_idx < num_classes) {
            output[batch_idx * num_classes + class_idx] = expf(input[batch_idx * num_classes + class_idx] - max_val) / exp_values[0];
        }
    }
}

#endif
