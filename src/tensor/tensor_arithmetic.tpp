#ifndef TENSOR_OPERATORS_TPP
#define TENSOR_OPERATORS_TPP

#include "../../inc/tensor/tensor.hpp"

#ifdef __NVCC__
#include "../../inc/device/tensor_cuda_wrappers.hpp"
#endif

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const Tensor<U>& other) const {
    if(this->device != other.device) {
        throw std::invalid_argument("Tensors must be on the same device for operations");
    }

#ifdef __NVCC__
    if(this->device == Device::GPU) {
        if(this->shape() == other.shape()) {
            return cuda_binary_op(other, cuda_ops::launch_tensor_add<T, U, std::common_type_t<T, U>>);
        } else if(this->can_broadcast(other)) {
            return cuda_broadcast_op(other, 0); // 0 = addition
        }
    }
#endif
    
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a + b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator-(const Tensor<U>& other) const {
    if (this->device != other.device) {
        throw std::invalid_argument("Tensors must be on the same device for operations");
    }

#ifdef __NVCC__
    if (this->device == Device::GPU) {
        if(this->shape() == other.shape()) {
            return cuda_binary_op(other, cuda_ops::launch_tensor_subtract<T, U, std::common_type_t<T, U>>);
        } else if(this->can_broadcast(other)) {
            return cuda_broadcast_op(other, 1); // 1 = subtraction
        }
    }
#endif

    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a - b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator*(const Tensor<U>& other) const {
    if (this->device != other.device) {
        throw std::invalid_argument("Tensors must be on the same device for operations");
    }

#ifdef __NVCC__
    if (this->device == Device::GPU) {
        if(this->shape() == other.shape()) {
            return cuda_binary_op(other, cuda_ops::launch_tensor_multiply<T, U, std::common_type_t<T, U>>);
        } else if(this->can_broadcast(other)) {
            return cuda_broadcast_op(other, 2); // 2 = multiplication
        }
    }
#endif

    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a * b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator/(const Tensor<U>& other) const {
    if (this->device != other.device) {
        throw std::invalid_argument("Tensors must be on the same device for operations");
    }

#ifdef __NVCC__
    if (this->device == Device::GPU) {
        if(this->shape() == other.shape()) {
            return cuda_binary_op(other, cuda_ops::launch_tensor_divide<T, U, std::common_type_t<T, U>>);
        } else if(this->can_broadcast(other)) {
            return cuda_broadcast_op(other, 3); // 3 = division
        }
    }
#endif

    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a / b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const U& scalar) const {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_scalar_op(scalar, cuda_ops::launch_tensor_scalar_add<T, U, std::common_type_t<T, U>>);
    }
#endif

    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a + b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator-(const U& scalar) const {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_scalar_op(scalar, cuda_ops::launch_tensor_scalar_subtract<T, U, std::common_type_t<T, U>>);
    }
#endif

    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a - b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator*(const U& scalar) const {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_scalar_op(scalar, cuda_ops::launch_tensor_scalar_multiply<T, U, std::common_type_t<T, U>>);
    }
#endif

    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a * b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator/(const U& scalar) const {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_scalar_op(scalar, cuda_ops::launch_tensor_scalar_divide<T, U, std::common_type_t<T, U>>);
    }
#endif

    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a / b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator+=(const Tensor<U>& other) {
#ifdef __NVCC__
    if (this->device == Device::GPU && this->device == other.device && this->shape() == other.shape()) {
        return cuda_inplace_tensor_op(other, cuda_ops::launch_tensor_inplace_add<T, U>);
    }
#endif

    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a += b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator-=(const Tensor<U>& other) {
#ifdef __NVCC__
    if (this->device == Device::GPU && this->device == other.device && this->shape() == other.shape()) {
        return cuda_inplace_tensor_op(other, cuda_ops::launch_tensor_inplace_subtract<T, U>);
    }
#endif

    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a -= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator*=(const Tensor<U>& other) {
#ifdef __NVCC__
    if (this->device == Device::GPU && this->device == other.device && this->shape() == other.shape()) {
        return cuda_inplace_tensor_op(other, cuda_ops::launch_tensor_inplace_multiply<T, U>);
    }
#endif

    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a *= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator/=(const Tensor<U>& other) {
#ifdef __NVCC__
    if (this->device == Device::GPU && this->device == other.device && this->shape() == other.shape()) {
        return cuda_inplace_tensor_op(other, cuda_ops::launch_tensor_inplace_divide<T, U>);
    }
#endif

    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a /= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator+=(const U& scalar) {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_inplace_scalar_op(scalar, cuda_ops::launch_tensor_inplace_scalar_add<T, U>);
    }
#endif

    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a += b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator-=(const U& scalar) {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_inplace_scalar_op(scalar, cuda_ops::launch_tensor_inplace_scalar_subtract<T, U>);
    }
#endif

    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a -= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator*=(const U& scalar) {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_inplace_scalar_op(scalar, cuda_ops::launch_tensor_inplace_scalar_multiply<T, U>);
    }
#endif

    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a *= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator/=(const U& scalar) {
#ifdef __NVCC__
    if (this->device == Device::GPU) {
        return cuda_inplace_scalar_op(scalar, cuda_ops::launch_tensor_inplace_scalar_divide<T, U>);
    }
#endif

    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a /= b; 
    });
}

#endif
