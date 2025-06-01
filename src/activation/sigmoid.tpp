#ifndef SIGMOID_TPP
#define SIGMOID_TPP

#include "../../inc/activation/sigmoid.hpp"

template <Numeric T>
Tensor<T> Sigmoid<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-input[i]));
    }

    return result;
}

template <Numeric T>
Tensor<T> Sigmoid<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = input[i] * (static_cast<T>(1) - input[i]);
    }

    return result;
}

#endif