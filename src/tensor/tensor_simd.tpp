#ifndef TENSOR_SIMD_TPP
#define TENSOR_SIMD_TPP

#include "../../inc/tensor/tensor.hpp"

#ifdef __NVCC__
#include "../../inc/device/tensor_cuda_wrappers.hpp"
#endif

template <Numeric T>
size_t Tensor<T>::get_broadcast_index(size_t i, const std::vector<size_t>& other_shape, const std::vector<size_t>& other_stride) const {
    size_t remaining = i;
    std::vector<size_t> indices(this->ndim());
    for(int dim = 0; dim < this->ndim(); ++dim) {
        indices[dim] = remaining / stride[dim];
        remaining %= stride[dim];
    }
    
    size_t other_idx = 0;
    for(int dim = 0; dim < this->ndim(); ++dim) {
        const size_t other_dim_idx = (other_shape[dim] == 1) ? 0 : indices[dim] % other_shape[dim];
        other_idx += other_dim_idx * other_stride[dim];
    }

    return other_idx;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::simd_with_tensor(
    const Tensor<U>& other, 
    std::function<void(std::common_type_t<T, U>&, const T&, const U&)> callback
) const {
    Tensor<std::common_type_t<T, U>> result(this->shape());

    if(this->shape() == other.shape()) {
        #pragma omp parallel for
        for(size_t i = 0; i < length(); ++i) {
            callback(result[i], data[i], other.data[i]);
        }
    } else if(other.is_scalar()) {
        #pragma omp parallel for
        for(size_t i = 0; i < length(); ++i) {
            callback(result[i], data[i], other.value());
        }
    } else if(this->is_scalar()) {
        #pragma omp parallel for
        for(size_t i = 0; i < other.length(); ++i) {
            callback(result[i], this->value(), other.data[i]);
        }
    } else {
        if(!this->can_broadcast(other)) {
            throw std::invalid_argument("Shapes cannot be broadcast together for operation");
        }
        
        std::vector<size_t> other_shape(this->ndim());
        std::vector<size_t> other_strides(this->ndim(), 1);

        for(int i = 0; i < this->ndim(); ++i) {
            if(i >= this->ndim() - other.ndim()) {
                other_shape[i] = other.shape(i - (this->ndim() - other.ndim()));
            } else {
                other_shape[i] = 1;
            }
        }
        
        for(int i = this->ndim() - 2; i >= 0; --i) {
            other_strides[i] = other_strides[i + 1] * other_shape[i + 1];
        }
        
        #pragma omp parallel for
        for(size_t i = 0; i < this->length(); ++i) {
            const size_t other_idx = get_broadcast_index(i, other_shape, other_strides);
            callback(result[i], data[i], other.data[other_idx]);
        }        
    }

    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::simd_with_scalar(
    const U& scalar, 
    std::function<void(std::common_type_t<T, U>&, const T&, const U&)> callback
) const {
    Tensor<std::common_type_t<T, U>> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < length(); ++i) {
        callback(result[i], data[i], scalar);
    }

    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::change_tensor_simd(
    const Tensor<U>& other, 
    std::function<void(T&, const U&)> callback
) {
    if(this->shape() == other.shape()) {
        #pragma omp parallel for
        for(size_t i = 0; i < length(); ++i) {
            callback(data[i], other.data[i]);
        }
    } else if(other.is_scalar()) {
        #pragma omp parallel for
        for(size_t i = 0; i < length(); ++i) {
            callback(data[i], other.value());
        }
    } else {
        if(!this->can_broadcast(other)) {
            throw std::invalid_argument("Shapes cannot be broadcast together for operation");
        }
        
        std::vector<size_t> other_strides(this->ndim(), 1);
        std::vector<size_t> other_shape(this->ndim());

        for(int i = 0; i < this->ndim(); ++i) {
            if(i >= this->ndim() - other.ndim()) {
                other_shape[i] = other.shape(i - (this->ndim() - other.ndim()));
            } else {
                other_shape[i] = 1;
            }
        }
        
        for(int i = this->ndim() - 2; i >= 0; --i) {
            other_strides[i] = other_strides[i + 1] * other_shape[i + 1];
        }
        
        #pragma omp parallel for
        for(size_t i = 0; i < this->length(); ++i) {
            const size_t other_idx = get_broadcast_index(i, other_shape, other_strides);
            callback(data[i], other.data[other_idx]);
        }        
    }

    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::change_tensor_scalar_simd(
    const U& scalar, 
    std::function<void(T&, const U&)> callback
) {
    #pragma omp parallel for
    for(size_t i = 0; i < length(); ++i) {
        callback(data[i], scalar);
    }

    return *this;
}

#ifdef __NVCC__

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::cuda_binary_op(
    const Tensor<U>& other,
    void (*cuda_kernel)(const T*, const U*, std::common_type_t<T, U>*, size_t)
) const {
    using R = std::common_type_t<T, U>;
    Tensor<R> result(this->shape());
    result.device = Device::GPU;
    
    if (result.owns_data) {
        delete[] result.data;
    }
    result.data = cuda_ops::cuda_malloc<R>(result.length());
    result.owns_data = true;
    
    cuda_kernel(this->data, other.data, result.data, this->length());
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::cuda_scalar_op(
    const U& scalar,
    void (*cuda_kernel)(const T*, U, std::common_type_t<T, U>*, size_t)
) const {
    using R = std::common_type_t<T, U>;
    Tensor<R> result(this->shape());
    result.device = this->device;
    
    if (result.owns_data) {
        delete[] result.data;
    }
    result.data = cuda_ops::cuda_malloc<R>(result.length());
    result.owns_data = true;
    
    cuda_kernel(this->data, scalar, result.data, this->length());
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::cuda_broadcast_op(
    const Tensor<U>& other,
    int operation
) const {
    using R = std::common_type_t<T, U>;
    
    // Determine result shape (broadcast to the larger tensor)
    Shape result_shape = this->shape();
    if (other.length() > this->length()) {
        result_shape = other.shape();
    }
    
    Tensor<R> result(result_shape);
    result.device = Device::GPU;
    
    if (result.owns_data) {
        delete[] result.data;
    }
    result.data = cuda_ops::cuda_malloc<R>(result.length());
    result.owns_data = true;
    
    // Calculate strides for broadcast
    const int result_ndim = result_shape.ndim();
    
    // Allocate device memory for strides and shapes
    int* d_a_strides = cuda_ops::cuda_malloc<int>(result_ndim);
    int* d_b_strides = cuda_ops::cuda_malloc<int>(result_ndim);
    int* d_result_strides = cuda_ops::cuda_malloc<int>(result_ndim);
    int* d_shape = cuda_ops::cuda_malloc<int>(result_ndim);
    
    // Prepare host arrays
    std::vector<int> a_strides(result_ndim, 0);
    std::vector<int> b_strides(result_ndim, 0);
    std::vector<int> result_strides(result_ndim);
    std::vector<int> shape_array(result_ndim);
    
    // Calculate result strides
    result_strides[result_ndim - 1] = 1;
    for (int i = result_ndim - 2; i >= 0; --i) {
        result_strides[i] = result_strides[i + 1] * result_shape[i + 1];
    }
    
    // Calculate strides for tensor a (this)
    const int a_offset = result_ndim - this->ndim();
    for (int i = 0; i < this->ndim(); ++i) {
        const int result_dim = a_offset + i;
        if (this->shape(i) == result_shape[result_dim]) {
            a_strides[result_dim] = this->stride[i];
        } else if (this->shape(i) == 1) {
            a_strides[result_dim] = 0; // Broadcasting dimension
        }
    }
    
    // Calculate strides for tensor b (other)
    const int b_offset = result_ndim - other.ndim();
    for (int i = 0; i < other.ndim(); ++i) {
        const int result_dim = b_offset + i;
        if (other.shape(i) == result_shape[result_dim]) {
            b_strides[result_dim] = other.stride[i];
        } else if (other.shape(i) == 1) {
            b_strides[result_dim] = 0; // Broadcasting dimension
        }
    }
    
    // Copy shape to array
    for (int i = 0; i < result_ndim; ++i) {
        shape_array[i] = result_shape[i];
    }
    
    // Copy to device
    cuda_ops::cuda_memcpy_host_to_device(d_a_strides, a_strides.data(), result_ndim);
    cuda_ops::cuda_memcpy_host_to_device(d_b_strides, b_strides.data(), result_ndim);
    cuda_ops::cuda_memcpy_host_to_device(d_result_strides, result_strides.data(), result_ndim);
    cuda_ops::cuda_memcpy_host_to_device(d_shape, shape_array.data(), result_ndim);
    
    // Launch broadcast kernel
    cuda_ops::launch_tensor_broadcast<T, U, R>(
        this->data, other.data, result.data,
        d_a_strides, d_b_strides, d_result_strides, d_shape,
        result.length(), result_ndim, operation
    );
    
    // Free device memory
    cuda_ops::cuda_free(d_a_strides);
    cuda_ops::cuda_free(d_b_strides);
    cuda_ops::cuda_free(d_result_strides);
    cuda_ops::cuda_free(d_shape);
    
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::cuda_inplace_tensor_op(
    const Tensor<U>& other,
    void (*cuda_kernel)(T*, const U*, size_t)
) {
    cuda_kernel(this->data, other.data, this->length());
    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::cuda_inplace_scalar_op(
    const U& scalar,
    void (*cuda_kernel)(T*, U, size_t)
) {
    cuda_kernel(this->data, scalar, this->length());
    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::cuda_comparison_op(
    const Tensor<U>& other,
    void (*cuda_kernel)(const T*, const U*, bool*, size_t)
) const {
    Tensor<bool> result(this->shape());
    result.device = Device::GPU;
    
    if (result.owns_data) {
        delete[] result.data;
    }
    result.data = cuda_ops::cuda_malloc<bool>(result.length());
    result.owns_data = true;
    
    cuda_kernel(this->data, other.data, result.data, this->length());
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::cuda_scalar_comparison_op(
    const U& scalar,
    void (*cuda_kernel)(const T*, U, bool*, size_t)
) const {
    Tensor<bool> result(this->shape());
    result.device = this->device;
    
    if (result.owns_data) {
        delete[] result.data;
    }
    result.data = cuda_ops::cuda_malloc<bool>(result.length());
    result.owns_data = true;
    
    cuda_kernel(this->data, scalar, result.data, this->length());
    return result;
}

template <Numeric T>
bool Tensor<T>::cuda_reduction_op(
    bool (*cuda_kernel)(const T*, size_t)
) const {
    return cuda_kernel(this->data, this->length());
}

#endif
#endif