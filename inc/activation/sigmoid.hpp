#ifndef SIGMOID_ACTIVATION_HPP
#define SIGMOID_ACTIVATION_HPP

#include <cmath>

#include "../abstract/activation.hpp"

template <Numeric T> class Sigmoid: public Activation<T> {
    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;

    public:
        friend std::ostream& operator<<(std::ostream& os, const Sigmoid<T>& sigmoid) {
            os << "Sigmoid()";
            return os;
        }
};

#include "../../src/activation/sigmoid.tpp"

#endif