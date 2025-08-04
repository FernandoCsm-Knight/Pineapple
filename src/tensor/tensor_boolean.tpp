#ifndef TENSOR_BOOLEAN_TPP
#define TENSOR_BOOLEAN_TPP

#include "../../inc/tensor/tensor.hpp"

#ifdef __NVCC__
#include "../../inc/device/tensor_cuda_wrappers.hpp"
#endif

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator==(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_equal<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a == b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator!=(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_not_equal<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a != b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_less_than<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a < b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<=(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_less_equal<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a <= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_greater_than<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a > b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>=(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_greater_equal<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a >= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator==(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_equal<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a == b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator!=(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_not_equal<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a != b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_less_than<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a < b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator<=(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_less_equal<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a <= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_greater_than<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a > b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator>=(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_greater_equal<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = (a >= b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator&&(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_logical_and<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) && static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator||(const Tensor<U>& other) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_comparison_op<U>(other, cuda_ops::launch_tensor_logical_or<T, U>);
    }
#endif
    return simd_with_tensor<U>(other, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) || static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator&&(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_logical_and<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) && static_cast<bool>(b);
    });
}

template <Numeric T>
template <Numeric U>
Tensor<bool> Tensor<T>::operator||(const U& scalar) const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        return cuda_scalar_comparison_op<U>(scalar, cuda_ops::launch_tensor_scalar_logical_or<T, U>);
    }
#endif
    return simd_with_scalar<U>(scalar, [](std::common_type_t<T, U>& result, const T& a, const U& b) {
        result = static_cast<bool>(a) || static_cast<bool>(b);
    });
}

template <Numeric T>
Tensor<bool> Tensor<T>::operator!() const {
#ifdef __NVCC__
    if (this->is_cuda()) {
        Tensor<bool> result(this->shape());
        result.current_device = Device::GPU;
        
        if (result.owns_data) {
            delete[] result.data;
        }
        result.data = cuda_ops::cuda_malloc<bool>(result.length());
        result.owns_data = true;
        
        cuda_ops::launch_tensor_logical_not<T>(this->data, result.data, this->length());
        return result;
    }
#endif
    return simd_with_scalar<bool>(true, [](std::common_type_t<T, bool>& result, const T& a, const bool&) {
        result = !static_cast<bool>(a);
    });
}

template <Numeric T>
bool Tensor<T>::any() const {
    bool result = false;

#ifdef __NVCC__
    if (this->is_cuda()) {
        result = cuda_reduction_op(cuda_ops::launch_tensor_any<T>);
    } else
#endif
    {
        for(size_t i = 0; !result && i < length(); ++i) {
            result = static_cast<bool>(data[i]);
        }
    }
    
    return result;
}

template <Numeric T>
bool Tensor<T>::all() const {
    bool result = true;

#ifdef __NVCC__
    if (this->is_cuda()) {
        result = cuda_reduction_op(cuda_ops::launch_tensor_all<T>);
    } else
#endif
    {
        for(size_t i = 0; result && i < length(); ++i) {
            result = static_cast<bool>(data[i]);
        }
    }

    return result;
}

#endif