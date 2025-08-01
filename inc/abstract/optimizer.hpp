#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../tensor/tensor.hpp"

template <Numeric T> class Optimizer {
    protected:
        T learning_rate;

    public:
        Optimizer(T lr): learning_rate(lr) {}
        virtual ~Optimizer() = default;

        virtual void to(Device target_device) = 0;
        virtual Optimizer<T>* copy() const = 0;

        virtual void step(
            Tensor<T>& weights, 
            const Tensor<T>& grad_weights, 
            Tensor<T>& bias, 
            const Tensor<T>& grad_bias
        ) = 0;
};

#endif