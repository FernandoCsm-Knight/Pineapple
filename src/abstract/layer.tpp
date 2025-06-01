#ifndef LAYER_TPP
#define LAYER_TPP

#include "../../inc/abstract/layer.hpp"

template <Numeric T>
bool Layer<T>::is_activation() const {
    return false;
} 

#endif