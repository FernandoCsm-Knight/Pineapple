#ifndef SOFTMAX_ACTIVATION_HPP
#define SOFTMAX_ACTIVATION_HPP

#include "../abstract/activation.hpp"

template <Numeric T> class Softmax: public Activation<T> {
    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;

    public:
        Softmax();

        friend std::ostream& operator<<(std::ostream& os, const Softmax<T>& softmax) {
            os << "Softmax()";
            return os;
        }
};

#include "../../src/activation/softmax.tpp"

#endif