#ifndef LEAKY_RELU_TPP
#define LEAKY_RELU_TPP

#include "../../inc/activation/leaky_relu.hpp"

template <Numeric T>
Tensor<T> LeakyReLU<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? input[i] : alpha * input[i];
    }

    return result;
}

template <Numeric T>
Tensor<T> LeakyReLU<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? static_cast<T>(1) : alpha;
    }

    return result;
}

#endif