#ifndef TENSOR_SIMD_TPP
#define TENSOR_SIMD_TPP

#include "../../inc/tensor/tensor.hpp"

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
        size_t other_dim_idx = (other_shape[dim] == 1) ? 0 : indices[dim] % other_shape[dim];
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
            size_t other_idx = get_broadcast_index(i, other_shape, other_strides);
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
            size_t other_idx = get_broadcast_index(i, other_shape, other_strides);
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

#endif