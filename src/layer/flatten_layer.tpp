#ifndef FLATTEN_LAYER_TPP
#define FLATTEN_LAYER_TPP

#include "../../inc/layer/flatten_layer.hpp"

template <Numeric T>
FlattenLayer<T>::FlattenLayer() : Layer<T>() {}

template <Numeric T>
Tensor<T> FlattenLayer<T>::forward(const Tensor<T>& input) {
    input_shape = input.shape();
    return input.flatten();
}

template <Numeric T>
Tensor<T> FlattenLayer<T>::backward(const Tensor<T>& grad_output) {
    return grad_output.reshape(input_shape);
}

#endif