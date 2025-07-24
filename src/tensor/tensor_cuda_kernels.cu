#ifndef TENSOR_CUDA_KERNELS_CU
#define TENSOR_CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <type_traits>

template <typename T, typename U, typename R, typename F>
__global__ void tensor_elementwise_kernel(const T* a, const U* b, R* result, size_t size, F operation) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = operation(a[idx], b[idx]);
}

template <typename T, typename U, typename R, typename F>
__global__ void tensor_scalar_kernel(const T* a, U scalar, R* result, size_t size, F operation) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = operation(a[idx], scalar);
}

template <typename T, typename U, typename F>
__global__ void tensor_inplace_elementwise_kernel(T* a, const U* b, size_t size, F operation) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) operation(a[idx], b[idx]);
}

template <typename T, typename U, typename F>
__global__ void tensor_inplace_scalar_kernel(T* a, U scalar, size_t size, F operation) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) operation(a[idx], scalar);
}

// Kernel para broadcast
template<typename T, typename U, typename R>
__global__ void tensor_broadcast_kernel(
    const T* a, const U* b, R* result,
    const size_t* a_strides, const size_t* b_strides,
    const size_t* result_strides, const int* shape,
    size_t total_elements, int ndim, int operation
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < total_elements) {
        size_t temp_idx = idx;
        size_t a_idx = 0, b_idx = 0;
        
        for (int i = 0; i < ndim; ++i) {
            size_t coord = temp_idx / result_strides[i];
            temp_idx %= result_strides[i];
            
            a_idx += coord * a_strides[i];
            b_idx += coord * b_strides[i];
        }
        
        const R val_a = static_cast<R>(a[a_idx]);
        const R val_b = static_cast<R>(b[b_idx]);
        
        switch (operation) {
            case 0: result[idx] = val_a + val_b; break;
            case 1: result[idx] = val_a - val_b; break;
            case 2: result[idx] = val_a * val_b; break;
            case 3: result[idx] = val_a / val_b; break;
        }
    }
}

// Kernel para fill
template<typename T>
__global__ void tensor_fill_kernel(T* data, T value, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) data[idx] = value;
}

// Kernel para cópia de dados
template<typename T>
__global__ void tensor_copy_kernel(const T* src, T* dst, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) dst[idx] = src[idx];
}

// Kernel para multiplicação de matrizes (dot product)
template<typename T, typename U, typename R>
__global__ void tensor_matmul_kernel(const T* a, const U* b, R* result, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N) {
        R sum = 0;
        
        for(int k = 0; k < K; ++k) {
            sum += static_cast<R>(a[row * K + k]) * static_cast<R>(b[k * N + col]);
        }

        result[row * N + col] = sum;
    }
}

// Kernels para operações de redução
template<typename T>
__global__ void tensor_min_kernel(const T* data, T* result, size_t size) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] : data[0];
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void tensor_max_kernel(const T* data, T* result, size_t size) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] : data[0];
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void tensor_sum_kernel(const T* data, T* result, size_t size) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] : T(0);
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

// Kernels para operações elemento por elemento
template<typename T>
__global__ void tensor_pow_kernel(const T* a, T* result, double exponent, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = pow(a[idx], exponent);
}

template<typename T>
__global__ void tensor_abs_kernel(const T* a, T* result, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = abs(a[idx]);
}

template<typename T>
__global__ void tensor_clip_kernel(const T* a, T* result, T min_val, T max_val, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        const T val = a[idx];
        result[idx] = (val < min_val) ? min_val : (val > max_val) ? max_val : val;
    }
}

template<typename T>
__global__ void tensor_normalize_kernel(const T* a, T* result, T min_val, T max_val, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = (a[idx] - min_val) / (max_val - min_val);
}

// Kernel para transpose
template<typename T>
__global__ void tensor_transpose_kernel(const T* a, T* result, int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < rows && col < cols) result[col * rows + row] = a[row * cols + col];
}

// Kernel para flip
template<typename T>
__global__ void tensor_flip_kernel(const T* a, T* result, const int* shape, const int* axes, 
                                  const int* strides, const int* result_strides, 
                                  int ndim, int num_axes, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size) {
        // Calculate multi-dimensional indices from linear index
        size_t temp_idx = idx;
        int indices[8]; // Assume max 8 dimensions
        
        for(int i = 0; i < ndim; ++i) {
            indices[i] = temp_idx / strides[i];
            temp_idx %= strides[i];
        }
        
        // Flip indices for specified axes
        for(int i = 0; i < num_axes; ++i) {
            int axis = axes[i];
            indices[axis] = shape[axis] - 1 - indices[axis];
        }
        
        // Calculate result linear index
        size_t result_idx = 0;
        for(int i = 0; i < ndim; ++i) {
            result_idx += indices[i] * result_strides[i];
        }
        
        result[result_idx] = a[idx];
    }
}

// Kernel para norm (euclidean)
template<typename T>
__global__ void tensor_norm_squared_kernel(const T* data, T* result, size_t size) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? data[i] * data[i] : T(0);
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

// Logical operation kernels

template<typename T>
__global__ void tensor_logical_not_kernel(const T* a, bool* result, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = !static_cast<bool>(a[idx]);
}

// Reduction kernels for any/all operations
template<typename T>
__global__ void tensor_any_kernel(const T* data, bool* result, size_t size) {
    extern __shared__ bool any_sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    any_sdata[tid] = (i < size) ? static_cast<bool>(data[i]) : false;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            any_sdata[tid] = any_sdata[tid] || any_sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = any_sdata[0];
}

template<typename T>
__global__ void tensor_all_kernel(const T* data, bool* result, size_t size) {
    extern __shared__ bool all_sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    all_sdata[tid] = (i < size) ? static_cast<bool>(data[i]) : true;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            all_sdata[tid] = all_sdata[tid] && all_sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = all_sdata[0];
}

// Variance kernel (two-pass algorithm)
template<typename T>
__global__ void tensor_variance_kernel(const T* data, T mean_val, T* result, size_t size) {
    extern __shared__ T var_sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        T diff = data[i] - mean_val;
        var_sdata[tid] = diff * diff;
    } else {
        var_sdata[tid] = T(0);
    }
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            var_sdata[tid] += var_sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = var_sdata[0];
}

// Argmax kernel
template<typename T>
__global__ void tensor_argmax_kernel(const T* data, int* result, size_t size) {
    extern __shared__ struct { T val; int idx; } argmax_sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        argmax_sdata[tid].val = data[i];
        argmax_sdata[tid].idx = i;
    } else {
        argmax_sdata[tid].val = (T)(-INFINITY);
        argmax_sdata[tid].idx = -1;
    }
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            if (argmax_sdata[tid + s].val > argmax_sdata[tid].val) {
                argmax_sdata[tid] = argmax_sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = argmax_sdata[0].idx;
}

// Slice kernel
template<typename T>
__global__ void tensor_slice_kernel(const T* data, T* result, int start, int step, size_t new_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < new_size) result[idx] = data[start + idx * step];
}

// Sqrt kernel (for std deviation)
template<typename T>
__global__ void tensor_sqrt_kernel(const T* data, T* result, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) result[idx] = sqrt(data[idx]);
}

#endif
