#ifndef TENSOR_BOOLEAN_TPP
#define TENSOR_BOOLEAN_TPP

#include "../../inc/tensor/tensor.hpp"

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator==(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a == b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator!=(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a != b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a < b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<=(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a <= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a > b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>=(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a >= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator==(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a == b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator!=(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a != b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a < b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<=(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a <= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a > b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>=(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a >= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator&&(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) && static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator||(const Tensor<U>& other) const {
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) || static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator&&(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) && static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator||(const U& scalar) const {
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) || static_cast<bool>(b);
    });
}

template <Numeric T>
Tensor<bool> Tensor<T>::operator!() const {
    return simd_with_scalar<bool>(true, [](std::common_type_t<T, bool>& result, const T& a, const bool&) {
        result = !static_cast<bool>(a);
    });
}

template <Numeric T>
bool Tensor<T>::any() const {
    for (size_t i = 0; i < length(); ++i) {
        if (static_cast<bool>(data[i])) {
            return true;
        }
    }
    return false;
}

template <Numeric T>
bool Tensor<T>::all() const {
    for (size_t i = 0; i < length(); ++i) {
        if (!static_cast<bool>(data[i])) {
            return false;
        }
    }
    return true;
}

#endif