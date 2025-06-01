#ifndef ACTIVATION_TPP
#define ACTIVATION_TPP

#include "../../inc/abstract/activation.hpp"

template <Numeric T>
Tensor<T> Activation<T>::forward(const Tensor<T>& input) {
    this->last_output = apply(input);
    return this->last_output;
}

template <Numeric T>
Tensor<T> Activation<T>::backward(const Tensor<T>& grad_output) {
    return last_activation ? grad_output : grad_output * derivative(this->last_output);
}

template <Numeric T>
bool Activation<T>::is_activation() const {
    return true;
}

template <Numeric T>
void Activation<T>::set_last_activation(bool is_last) {
    last_activation = is_last;
}

template <Numeric T>
bool Activation<T>::is_last_activation() const {
    return last_activation;
}

#endif