#ifndef SGD_TPP
#define SGD_TPP

#include "../../inc/optimizer/sgd.hpp"

template <Numeric T>
SGD<T>::SGD(T learning_rate, T momentum): Optimizer<T>(learning_rate), momentum(momentum) {
    weight_v = nullptr;
    bias_v = nullptr;
}

template <Numeric T>
SGD<T>::~SGD() {
    delete weight_v;
    delete bias_v;
}

template <Numeric T>
Optimizer<T>* SGD<T>::copy() const {
    return new SGD<T>(this->learning_rate, momentum);
}

template <Numeric T>
void SGD<T>::step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias) {
    if (weight_v == nullptr) {
        weight_v = new Tensor<T>(weights.shape(), 0);
        bias_v = new Tensor<T>(bias.shape(), 0);
    }

    *weight_v = *weight_v * momentum - grad_weights * this->learning_rate;
    *bias_v = *bias_v * momentum - grad_bias * this->learning_rate;

    weights -= *weight_v;
    bias -= *bias_v;
}

#endif