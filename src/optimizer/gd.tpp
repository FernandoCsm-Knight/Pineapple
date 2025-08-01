#ifndef GD_TPP
#define GD_TPP

#include "../../inc/optimizer/gd.hpp"

template <Numeric T>
GD<T>::GD(T learning_rate): Optimizer<T>(learning_rate) {}

template <Numeric T>
void GD<T>::to(Device target_device) {}

template <Numeric T>
Optimizer<T>* GD<T>::copy() const {
    return new GD<T>(this->learning_rate);
}

template <Numeric T>
void GD<T>::step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias) {
    weights -= grad_weights * this->learning_rate;
    bias -= grad_bias * this->learning_rate;
}

#endif