#ifndef ADAMW_TPP
#define ADAMW_TPP

#include "../../inc/optimizer/adamw.hpp"

template <Numeric T>
AdamW<T>::AdamW(T learning_rate, T weight_decay, T beta1, T beta2, T epsilon)
    : Optimizer<T>(learning_rate), weight_decay(weight_decay), 
    beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {
    weight_m = nullptr;
    weight_v = nullptr;
    bias_m = nullptr;
    bias_v = nullptr;
}

template <Numeric T>
AdamW<T>::~AdamW() {
    delete weight_m;
    delete weight_v;
    delete bias_m;
    delete bias_v;
}

template <Numeric T>
void AdamW<T>::step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) {
    t++;
    
    if (weight_m == nullptr) {
        weight_m = new Tensor<T>(weights.shape(), 0);
        weight_v = new Tensor<T>(weights.shape(), 0);
        bias_m = new Tensor<T>(bias.shape(), 0);
        bias_v = new Tensor<T>(bias.shape(), 0);
    }
    
    T beta1_t = std::pow(beta1, t);
    T beta2_t = std::pow(beta2, t);
    T alpha_t = this->learning_rate * std::sqrt(1 - beta2_t) / (1 - beta1_t);
    
    weights *= (1 - this->learning_rate * weight_decay);
    
    *weight_m = *weight_m * beta1 + grad_weights * (1 - beta1);
    
    Tensor<T> weight_grad_squared = grad_weights * grad_weights;
    *weight_v = *weight_v * beta2 + weight_grad_squared * (1 - beta2);
    
    #pragma omp parallel for
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] -= alpha_t * (*weight_m)[i] / (std::sqrt((*weight_v)[i]) + epsilon);
    }
    
    *bias_m = *bias_m * beta1 + grad_bias * (1 - beta1);
    
    Tensor<T> bias_grad_squared = grad_bias * grad_bias;
    *bias_v = *bias_v * beta2 + bias_grad_squared * (1 - beta2);
    
    #pragma omp parallel for
    for (size_t i = 0; i < bias.length(); ++i) {
        bias[i] -= alpha_t * (*bias_m)[i] / (std::sqrt((*bias_v)[i]) + epsilon);
    }
}

#endif