#ifndef TENSOR_OPERATORS_TPP
#define TENSOR_OPERATORS_TPP

#include "../../inc/tensor/tensor.hpp"

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a + b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator-(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a - b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator*(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a * b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator/(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a / b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator+(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a + b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator-(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a - b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator*(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a * b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::operator/(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) { 
        result = a / b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator+=(const Tensor<U>& other) {
    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a += b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator-=(const Tensor<U>& other) {
    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a -= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator*=(const Tensor<U>& other) {
    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a *= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator/=(const Tensor<U>& other) {
    return change_tensor_simd<U>(other, [](T& a, const U& b) { 
        a /= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator+=(const U& scalar) {
    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a += b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator-=(const U& scalar) {
    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a -= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator*=(const U& scalar) {
    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a *= b; 
    });
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator/=(const U& scalar) {
    return change_tensor_scalar_simd<U>(scalar, [](T& a, const U& b) { 
        a /= b; 
    });
}

#endif
