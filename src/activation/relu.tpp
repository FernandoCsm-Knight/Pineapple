#ifndef RELU_TPP
#define RELU_TPP

#include "../../inc/activation/relu.hpp"

template <Numeric T>
Tensor<T> ReLU<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = std::max(static_cast<T>(0), input[i]);
    }

    return result;
}

template <Numeric T>
Tensor<T> ReLU<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
    }

    return result;
}


#endif 