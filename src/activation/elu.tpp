#ifndef ELU_TPP
#define ELU_TPP

#include "../../inc/activation/elu.hpp"

template <Numeric T>
Tensor<T> ELU<T>::apply(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? 
                    input[i] : 
                    alpha * (std::exp(input[i]) - static_cast<T>(1));
    }

    return result;
}

template <Numeric T>
Tensor<T> ELU<T>::derivative(const Tensor<T>& input) const {
    Tensor<T> result(input.shape());

    #pragma omp parallel for
    for(size_t i = 0; i < input.length(); ++i) {
        result[i] = (input[i] > static_cast<T>(0)) ? 
                    static_cast<T>(1) : 
                    alpha * std::exp(input[i]);
    }

    return result;
}

#endif