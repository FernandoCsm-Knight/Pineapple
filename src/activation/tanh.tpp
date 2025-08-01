#ifndef TANH_TPP
#define TANH_TPP

#include "../../inc/activation/tanh.hpp"

#ifdef __NVCC__
#include "../../inc/device/activation_cuda_wrappers.hpp"
#endif

template <Numeric T>
Tensor<T> Tanh<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    result.to(input.get_device());

    #ifdef __NVCC__
    if(input.is_cuda()) {
        cuda_activation_ops::launch_tanh_apply(input.data_ptr(), result.data_ptr(), input.length());
    } else 
    #endif
    {
        #pragma omp parallel for
        for(size_t i = 0; i < input.length(); ++i) {
            result[i] = std::tanh(input[i]);
        }
    }

    return result;
}

template <Numeric T>
Tensor<T> Tanh<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    result.to(input.get_device());

    #ifdef __NVCC__
    if(input.is_cuda()) {
        cuda_activation_ops::launch_tanh_derivative(input.data_ptr(), result.data_ptr(), input.length());
    } else {
    #endif
        #pragma omp parallel for
        for(size_t i = 0; i < input.length(); ++i) {
            result[i] = static_cast<T>(1) - input[i] * input[i];
        }
    #ifdef __NVCC__
    }
    #endif

    return result;
}

#endif