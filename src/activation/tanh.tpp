#ifndef TANH_TPP
#define TANH_TPP

#include "../../inc/activation/tanh.hpp"

template <Numeric T>
Tensor<T> Tanh<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = std::tanh(input[i]);
    }

    return result;
}

template <Numeric T>
Tensor<T> Tanh<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = static_cast<T>(1) - input[i] * input[i];
    }

    return result;
}

#endif