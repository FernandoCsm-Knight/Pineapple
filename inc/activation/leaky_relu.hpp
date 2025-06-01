#ifndef LEAKY_RELU_ACTIVATION_HPP
#define LEAKY_RELU_ACTIVATION_HPP

#include "../abstract/activation.hpp"

template <Numeric T> class LeakyReLU: public Activation<T> {
    private:
        T alpha;
        
    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;

    public:
        explicit LeakyReLU(T alpha = 0.01) : alpha(alpha) {}

        friend std::ostream& operator<<(std::ostream& os, const LeakyReLU<T>& leaky_relu) {
            os << "LeakyReLU(alpha=" << leaky_relu.alpha << ")";
            return os;
        }
};

#include "../../src/activation/leaky_relu.tpp"

#endif