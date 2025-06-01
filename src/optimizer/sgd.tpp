#ifndef SGD_TPP
#define SGD_TPP

#include "../../inc/optimizer/sgd.hpp"

template <Numeric T>
SGD<T>::SGD(T learning_rate, T momentum, bool regression): Optimizer<T>(learning_rate), momentum(momentum) {
    weight_v = nullptr;
    bias_v = nullptr;
    this->regression = false;
}

template <Numeric T>
SGD<T>::~SGD() {
    delete weight_v;
    delete bias_v;
}

template <Numeric T>
void SGD<T>::step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) {
    if (weight_v == nullptr) {
        weight_v = new Tensor<T>(weights.shape(), 0);
        bias_v = new Tensor<T>(bias.shape(), 0);
    }

    if(this->regression) {
        *weight_v = *weight_v * momentum - grad_weights * this->learning_rate / batch_size;
        *bias_v = *bias_v * momentum - grad_bias * this->learning_rate / batch_size;
    } else {
        *weight_v = *weight_v * momentum - grad_weights * this->learning_rate;
        *bias_v = *bias_v * momentum - grad_bias * this->learning_rate;
    }

    weights += *weight_v;
    bias += *bias_v;
}

#endif