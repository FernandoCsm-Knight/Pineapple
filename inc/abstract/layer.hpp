#ifndef LAYER_HPP
#define LAYER_HPP

#include "../types/numeric.hpp"

template <Numeric T> class Layer {
    public:
        virtual ~Layer() = default;

        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

        virtual bool is_activation() const;
};

#include "../../src/abstract/layer.tpp"

#endif 