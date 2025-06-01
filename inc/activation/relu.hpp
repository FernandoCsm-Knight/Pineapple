#ifndef RELU_ACTIVATION_HPP
#define RELU_ACTIVATION_HPP

#include "../abstract/activation.hpp"

template <Numeric T> class ReLU: public Activation<T> {
    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;

    public:
        friend std::ostream& operator<<(std::ostream& os, const ReLU<T>& relu) {
            os << "ReLU()";
            return os;
        }
};

#include "../../src/activation/relu.tpp"

#endif