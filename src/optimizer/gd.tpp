#ifndef GD_TPP
#define GD_TPP

#include "../../inc/optimizer/gd.hpp"

template <Numeric T>
GD<T>::GD(T learning_rate, bool regression): Optimizer<T>(learning_rate) { this->regression = regression;}

template <Numeric T>
void GD<T>::step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) {
    if(regression) {
        weights -= grad_weights * this->learning_rate / batch_size;
        bias -= grad_bias * this->learning_rate / batch_size;
    } else {
        weights -= grad_weights * this->learning_rate;
        bias -= grad_bias * this->learning_rate;
    }
}

#endif