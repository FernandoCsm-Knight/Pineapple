#ifndef RELU_TPP
#define RELU_TPP

#include "../../inc/activation/relu.hpp"

#ifdef __NVCC__
#include "../../inc/device/activation_cuda_wrappers.hpp"
#endif

template <Numeric T>
Tensor<T> ReLU<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    result.to(input.device());

    #ifdef __NVCC__
    if(input.is_cuda()) {
        cuda_activation_ops::launch_relu_apply(input.data_ptr(), result.data_ptr(), input.length());
    } else {
    #endif
        #pragma omp parallel for
        for(size_t i = 0; i < input.length(); ++i) {
            result[i] = std::max(static_cast<T>(0), input[i]);
        }
    #ifdef __NVCC__
    }
    #endif

    return result;
}

template <Numeric T>
Tensor<T> ReLU<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    result.to(input.device());

    #ifdef __NVCC__
    if(input.is_cuda()) {
        cuda_activation_ops::launch_relu_derivative(input.data_ptr(), result.data_ptr(), input.length());
    } else 
    #endif
    {
        #pragma omp parallel for
        for(size_t i = 0; i < input.length(); ++i) {
            result[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }

    return result;
}


#endif 