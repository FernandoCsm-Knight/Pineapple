#ifndef SOFTMAX_TPP
#define SOFTMAX_TPP

#include "../../inc/activation/softmax.hpp"

#ifdef __NVCC__
#include "../../inc/device/activation_cuda_wrappers.hpp"
#include "../../inc/device/tensor_cuda_wrappers.hpp"
#endif

template <Numeric T>
Softmax<T>::Softmax(): Activation<T>() {
    this->last_activation = true;
}

template <Numeric T> 
Tensor<T> Softmax<T>::apply(const Tensor<T>& input) const  {
    Tensor<T> result(input.shape());
    result.to(input.device());
    
    #ifdef __NVCC__
    if(input.is_cuda()) {
        int batch_size = input.shape(0);
        int num_classes = input.shape(1);
        cuda_activation_ops::launch_softmax_apply(input.data_ptr(), result.data_ptr(), batch_size, num_classes, input.length());
    } else 
    #endif
    {
        #pragma omp parallel for
        for (int i = 0; i < input.shape(0); ++i) {
            T max = input(i, 0).value();
            for (int j = 1; j < input.shape(1); ++j) {
                max = std::max(max, input(i, j).value());
            }
            
            T sum = 0;
            for (int j = 0; j < input.shape(1); ++j) {
                T exp = std::exp(input(i, j).value() - max);
                result(i, j) = exp;
                sum += exp;
            }
            
            for (int j = 0; j < input.shape(1); ++j) {
                result(i, j) = result(i, j).value() / sum;
            }
        }
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Softmax<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());
    result.to(input.device());

    #ifdef __NVCC__
    if(input.is_cuda()) {
        cuda_ops::launch_tensor_copy(input.data_ptr(), result.data_ptr(), input.length());
    } else {
    #endif
        result = input; // Identity for softmax derivative (handled in loss function)
    #ifdef __NVCC__
    }
    #endif

    return result;
}

#endif