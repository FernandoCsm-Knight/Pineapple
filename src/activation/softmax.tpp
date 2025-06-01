#ifndef SOFTMAX_TPP
#define SOFTMAX_TPP

#include "../../inc/activation/softmax.hpp"

template <Numeric T>
Softmax<T>::Softmax(): Activation<T>() {
    this->last_activation = true;
}

template <Numeric T> 
Tensor<T> Softmax<T>::apply(const Tensor<T>& input) const  {
    Tensor<T> result(input.shape());
    
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
    
    return result;
}

template <Numeric T>
Tensor<T> Softmax<T>::derivative(const Tensor<T>& input) const {
    return input;
}

#endif