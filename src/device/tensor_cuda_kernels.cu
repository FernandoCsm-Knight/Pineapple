#ifndef TENSOR_CUDA_KERNELS_CU
#define TENSOR_CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <type_traits>

extern __shared__ unsigned char shared_data[];

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
    const int* a_strides, const int* b_strides,
    const int* result_strides, const int* shape,
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

// Kernel para conversão de tipo
template<typename T, typename U>
__global__ void tensor_type_convert_kernel(const U* src, T* dst, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) dst[idx] = static_cast<T>(src[idx]);
}

// Kernels para operações de redução
template<typename T>
__global__ void tensor_min_kernel(const T* data, T* result, size_t size) {
    T* sdata = reinterpret_cast<T*>(shared_data);
    
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
    T* sdata = reinterpret_cast<T*>(shared_data);
    
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
    T* sdata = reinterpret_cast<T*>(shared_data);
    
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
        size_t temp_idx = idx;
        int indices[8];
        
        for(int i = 0; i < ndim; ++i) {
            indices[i] = temp_idx / strides[i];
            temp_idx %= strides[i];
        }
        
        for(int i = 0; i < num_axes; ++i) {
            int axis = axes[i];
            indices[axis] = shape[axis] - 1 - indices[axis];
        }
        
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
    T* sdata = reinterpret_cast<T*>(shared_data);
    
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
    bool* sdata = reinterpret_cast<bool*>(shared_data);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? static_cast<bool>(data[i]) : false;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] = sdata[tid] || sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

template<typename T>
__global__ void tensor_all_kernel(const T* data, bool* result, size_t size) {
    bool* sdata = reinterpret_cast<bool*>(shared_data);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? static_cast<bool>(data[i]) : true;
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] = sdata[tid] && sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
}

// Variance kernel (two-pass algorithm)
template<typename T>
__global__ void tensor_variance_kernel(const T* data, T mean_val, T* result, size_t size) {
    T* sdata = reinterpret_cast<T*>(shared_data);
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        T diff = data[i] - mean_val;
        sdata[tid] = diff * diff;
    } else {
        sdata[tid] = T(0);
    }
    __syncthreads();
    
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s && (i + s) < size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if(tid == 0) result[blockIdx.x] = sdata[0];
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

// Dilate kernel
template<typename T>
__global__ void tensor_dilate_kernel(const T* input, T* output, 
                                    const int* input_shape, const int* output_shape,
                                    const int* input_strides, const int* output_strides,
                                    const int* axes, int num_axes, int dilation_size,
                                    int ndim, size_t output_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < output_size) {
        size_t temp_idx = idx;
        int output_indices[8];
        
        for(int i = 0; i < ndim; ++i) {
            output_indices[i] = temp_idx / output_strides[i];
            temp_idx %= output_strides[i];
        }
        
        bool is_data_position = true;
        int input_indices[8];
        
        for(int i = 0; i < ndim; ++i) {
            bool is_dilated_axis = false;
            for(int j = 0; j < num_axes; ++j) {
                if(axes[j] == i) {
                    is_dilated_axis = true;
                    break;
                }
            }
            
            if(is_dilated_axis) {
                if(output_indices[i] % (dilation_size + 1) != 0) {
                    is_data_position = false;
                    break;
                }
                input_indices[i] = output_indices[i] / (dilation_size + 1);
            } else {
                input_indices[i] = output_indices[i];
            }
        }
        
        if(is_data_position) {
            size_t input_idx = 0;
            for(int i = 0; i < ndim; ++i) {
                input_idx += input_indices[i] * input_strides[i];
            }
            output[idx] = input[input_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

// Pad kernel
template<typename T>
__global__ void tensor_pad_kernel(const T* input, T* output,
                                 const int* input_shape, const int* output_shape,
                                 const int* input_strides, const int* output_strides,
                                 const int* axes, int num_axes, int pad_size,
                                 int ndim, size_t output_size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < output_size) {
        size_t temp_idx = idx;
        int output_indices[8];
        
        for(int i = 0; i < ndim; ++i) {
            output_indices[i] = temp_idx / output_strides[i];
            temp_idx %= output_strides[i];
        }
        
        bool is_input_region = true;
        int input_indices[8];
        
        for(int i = 0; i < ndim; ++i) {
            bool is_padded_axis = false;
            for(int j = 0; j < num_axes; ++j) {
                if(axes[j] == i) {
                    is_padded_axis = true;
                    break;
                }
            }
            
            if(is_padded_axis) {
                if(output_indices[i] < pad_size || output_indices[i] >= input_shape[i] + pad_size) {
                    is_input_region = false;
                    break;
                }
                input_indices[i] = output_indices[i] - pad_size;
            } else {
                input_indices[i] = output_indices[i];
            }
        }
        
        if(is_input_region) {
            size_t input_idx = 0;
            for(int i = 0; i < ndim; ++i) {
                input_idx += input_indices[i] * input_strides[i];
            }
            output[idx] = input[input_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

template<typename T, typename U, typename R>
__global__ void tensor_dot_kernel(const T* a, const U* b, R* result, 
                                 int a_rows, int a_cols, int b_rows, int b_cols,
                                 int result_rows, int result_cols, 
                                 bool a_is_vector, bool b_is_vector) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(a_is_vector && b_is_vector) {
        // Vector dot vector (scalar result)
        if(row == 0 && col == 0) {
            R sum = 0;
            for(int k = 0; k < a_cols; ++k) {
                sum += static_cast<R>(a[k]) * static_cast<R>(b[k]);
            }
            result[0] = sum;
        }
    } else if(a_is_vector && !b_is_vector) {
        // Vector x Matrix: (1, n) x (n, p) = (1, p)
        if(row == 0 && col < result_cols) {
            R sum = 0;
            for(int k = 0; k < a_cols; ++k) {
                sum += static_cast<R>(a[k]) * static_cast<R>(b[k * b_cols + col]);
            }
            result[col] = sum;
        }
    } else if(!a_is_vector && b_is_vector) {
        // Matrix x Vector: (m, n) x (n, 1) = (m, 1)
        if(col == 0 && row < result_rows) {
            R sum = 0;
            for(int k = 0; k < a_cols; ++k) {
                sum += static_cast<R>(a[row * a_cols + k]) * static_cast<R>(b[k]);
            }
            result[row] = sum;
        }
    } else {
        // Matrix x Matrix: (m, n) x (n, p) = (m, p)
        if(row < result_rows && col < result_cols) {
            R sum = 0;
            for(int k = 0; k < a_cols; ++k) {
                sum += static_cast<R>(a[row * a_cols + k]) * static_cast<R>(b[k * b_cols + col]);
            }
            result[row * result_cols + col] = sum;
        }
    }
}

#endif
