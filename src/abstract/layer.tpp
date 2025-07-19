#ifndef LAYER_TPP
#define LAYER_TPP

#include "../../inc/abstract/layer.hpp"

template <Numeric T>
Layer<T>::~Layer() {
    if(this->optimizer) delete this->optimizer;
}

template <Numeric T>
void Layer<T>::train() {
   training = true;
}

template <Numeric T>
void Layer<T>::eval() {
    training = false;
}

template <Numeric T>
bool Layer<T>::is_in_train_mode() {
    return training;
}

template <Numeric T>
bool Layer<T>::is_activation() const {
    return false;
} 

template <Numeric T>
bool Layer<T>::is_optimization() const {
    return true;
}

template <Numeric T>
void Layer<T>::set_optimizer(Optimizer<T>* optim) {
    if(this->is_optimization()) {
        if(this->optimizer) delete this->optimizer;
        this->optimizer = optim->copy();
    }
}

#endif