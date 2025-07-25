#ifndef LINEARLAYER_TPP 
#define LINEARLAYER_TPP

#include "../../inc/layer/linear_layer.hpp"

#include <random>

template <Numeric T>
LinearLayer<T>::LinearLayer(int in_features, int out_features)
    : in_features(in_features), out_features(out_features) {
    
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
Tensor<T> LinearLayer<T>::forward(const Tensor<T>& input) {
    if (input.shape(input.ndim() - 1) != in_features) {
        throw std::invalid_argument("Input features don't match layer's in_features");
    }
    
    this->last_input = input;
    return input.dot(weights) + bias;
}

template <Numeric T>
Tensor<T> LinearLayer<T>::backward(const Tensor<T>& grad_weights) {
    if(!this->optimizer) {
        throw std::runtime_error("Optimizer not set for LinearLayer");
    }

    Tensor<T> previous_grad = grad_weights.dot(weights.transpose());
    const int batch_size = grad_weights.shape(0);

    this->optimizer->step(
        weights,
        last_input.transpose().dot(grad_weights) / batch_size, 
        bias,
        grad_weights.sum(0, true) / batch_size
    );

    return previous_grad;
}

#endif