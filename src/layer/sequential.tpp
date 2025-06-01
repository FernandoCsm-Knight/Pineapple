#ifndef SEQUENTIAL_TPP
#define SEQUENTIAL_TPP

#include "../../inc/layer/sequential.hpp"

template <Numeric T>
Sequential<T>::Sequential(std::initializer_list<Layer<T>*> layer_list) {
    layers = new Layer<T>*[layer_list.size()];
    size = static_cast<int>(layer_list.size());

    int i = 0;
    for (auto layer : layer_list) {
        layers[i++] = layer;
    }
}

template <Numeric T>
Sequential<T>::~Sequential() {
    for (int i = 0; i < size; ++i) {
        delete layers[i];
    }

    delete[] layers;
}

template <Numeric T>
Tensor<T> Sequential<T>::forward(const Tensor<T>& input) {
    Tensor<T> output = input;

    for(int i = 0; i < size; ++i) {
        output = layers[i]->forward(output);
    }

    return output;
}

template <Numeric T>
Tensor<T> Sequential<T>::backward(const Tensor<T>& grad_output) {
    Tensor<T> grad = grad_output;

    for(int i = size - 1; i >= 0; --i) {
        grad = layers[i]->backward(grad);
    }

    return grad;
}

template <Numeric T>
Layer<T>* Sequential<T>::first() const {
    return layers[0];
}

template <Numeric T>
Layer<T>* Sequential<T>::last() const {
    return layers[size - 1];
}

template <Numeric T>
Layer<T>* Sequential<T>::operator[](int index) const {
    if(index < 0 || index >= size) {
        throw std::out_of_range("Index out of range");
    }

    return layers[index];
}

template <Numeric T>
bool Sequential<T>::length() const {
    return size;
}

template <Numeric T>
bool Sequential<T>::empty() const {
    return size == 0;
}

template <Numeric T>
Layer<T>** Sequential<T>::begin() {
    return layers;
}

template <Numeric T>
Layer<T>** Sequential<T>::end() {
    return layers + size;
}

template <Numeric T>
const Layer<T>** Sequential<T>::begin() const {
    return layers;
}

template <Numeric T>
const Layer<T>** Sequential<T>::end() const {
    return layers + size;
}

#endif
