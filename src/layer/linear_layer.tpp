#ifndef LINEARLAYER_TPP 
#define LINEARLAYER_TPP

#include "../../inc/layer/linear_layer.hpp"

#include <random>

template <Numeric T>
LinearLayer<T>::LinearLayer(int in_features, int out_features, Optimizer<T>* optim)
    : in_features(in_features), out_features(out_features), optimizer(optim) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(2.0 / in_features));
    
    weights = Tensor<T>(in_features, out_features);
    
    #pragma omp parallel for
    for (size_t i = 0; i < weights.length(); ++i) {
        weights[i] = dist(gen);
    }

    bias = Tensor<T>(Shape(1, out_features), static_cast<T>(1));
}

template <Numeric T>
LinearLayer<T>::~LinearLayer() {
    delete optimizer;
}

template <Numeric T>
Tensor<T> LinearLayer<T>::forward(const Tensor<T>& input) {
    if (input.shape(input.ndim() - 1) != in_features) {
        throw std::invalid_argument("Input features don't match layer's in_features");
    }
    
    this->last_input = input;
    return input.dot(weights) + bias;
}

template <Numeric T>
Tensor<T> LinearLayer<T>::backward(const Tensor<T>& grad_weights) {
    Tensor<T> previous_grad = grad_weights.dot(weights.transpose());
    int batch_size = grad_weights.shape(0);

    optimizer->step(
        weights,
        last_input.transpose().dot(grad_weights), 
        bias,
        grad_weights.sum(0, true),
        batch_size
    );

    return previous_grad;
}

#endif