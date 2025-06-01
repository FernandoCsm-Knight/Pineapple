#ifndef TANH_ACTIVATION_HPP
#define TANH_ACTIVATION_HPP

#include <cmath>

#include "../abstract/activation.hpp"

template <Numeric T> class Tanh: public Activation<T> {
    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;

    public:
        friend std::ostream& operator<<(std::ostream& os, const Tanh<T>& tanh) {
            os << "Tanh()";
            return os;
        }
};

#include "../../src/activation/tanh.tpp"

#endif