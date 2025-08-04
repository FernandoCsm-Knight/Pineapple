#ifndef LOSS_CUDA_KERNELS_CU
#define LOSS_CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "../../inc/device/cuda_macros.hpp"

// Binary Cross Entropy Loss Kernels
template<typename T>
__global__ void binary_cross_entropy_compute_kernel(const T* predictions, const T* targets, T* losses, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T pred = fmaxf(fminf(predictions[idx], static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
        const T target = targets[idx];
        losses[idx] = -(target * logf(pred) + (static_cast<T>(1) - target) * logf(static_cast<T>(1) - pred));
    }
}

template<typename T>
__global__ void binary_cross_entropy_gradient_kernel(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T pred = fmaxf(fminf(predictions[idx], static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
        const T target = targets[idx];
        grad[idx] = (-target / pred + (static_cast<T>(1) - target) / (static_cast<T>(1) - pred)) / static_cast<T>(batch_size);
    }
}

// Cross Entropy Loss Kernels
template<typename T>
__global__ void cross_entropy_compute_kernel(const T* predictions, const T* targets, T* losses, size_t batch_size, size_t num_classes) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const int target_idx = static_cast<int>(targets[idx]);
        const T pred_prob = fmaxf(predictions[idx * num_classes + target_idx], static_cast<T>(1e-7));
        losses[idx] = -logf(pred_prob);
    }
}

template<typename T>
__global__ void cross_entropy_gradient_kernel(const T* predictions, const T* targets, T* grad, size_t batch_size, size_t num_classes) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_elements = batch_size * num_classes;
    
    if(idx < total_elements) {
        const size_t batch_idx = idx / num_classes;
        const size_t class_idx = idx % num_classes;
        
        const int target_idx = static_cast<int>(targets[batch_idx]);
        
        grad[idx] = predictions[idx];
        if(class_idx == target_idx) {
            grad[idx] -= static_cast<T>(1);
        }
        grad[idx] /= static_cast<T>(batch_size);
    }
}

// MSE Loss Kernels
template<typename T>
__global__ void mse_compute_kernel(const T* predictions, const T* targets, T* losses, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T diff = predictions[idx] - targets[idx];
        losses[idx] = diff * diff;
    }
}

template<typename T>
__global__ void mse_gradient_kernel(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        grad[idx] = static_cast<T>(2) * (predictions[idx] - targets[idx]) / static_cast<T>(batch_size);
    }
}

// MAE Loss Kernels
template<typename T>
__global__ void mae_compute_kernel(const T* predictions, const T* targets, T* losses, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) losses[idx] = fabsf(predictions[idx] - targets[idx]);
}

template<typename T>
__global__ void mae_gradient_kernel(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T diff = predictions[idx] - targets[idx];
        grad[idx] = (diff > static_cast<T>(0) ? static_cast<T>(1) : 
                    (diff < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(0))) / static_cast<T>(batch_size);
    }
}

// Huber Loss Kernels
template<typename T>
__global__ void huber_compute_kernel(const T* predictions, const T* targets, T* losses, T delta, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T diff = fabsf(predictions[idx] - targets[idx]);
        if(diff < delta) {
            losses[idx] = static_cast<T>(0.5) * diff * diff;
        } else {
            losses[idx] = delta * (diff - static_cast<T>(0.5) * delta);
        }
    }
}

template<typename T>
__global__ void huber_gradient_kernel(const T* predictions, const T* targets, T* grad, T delta, size_t batch_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < batch_size) {
        const T diff = predictions[idx] - targets[idx];
        const T abs_diff = fabsf(diff);
        
        if(abs_diff < delta) {
            grad[idx] = diff / static_cast<T>(batch_size);
        } else {
            grad[idx] = delta * (diff > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1)) / static_cast<T>(batch_size);
        }
    }
}

// Reduction kernel for computing mean loss
template<typename T>
__global__ void reduction_sum_kernel(const T* input, T* output, size_t size) {
    T* sdata = reinterpret_cast<T*>(shared_data);
    
    const size_t tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : static_cast<T>(0);
    __syncthreads();
    
    for(size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if(tid == 0) output[blockIdx.x] = sdata[0];
}

#endif
