#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../tensor/tensor.hpp"

template <Numeric T> class Optimizer {
    protected:
        T learning_rate;

    public:
        Optimizer(T lr): learning_rate(lr) {}
        virtual ~Optimizer() = default;

        virtual void step(
            Tensor<T>& weights, 
            const Tensor<T>& grad_weights, 
            Tensor<T>& bias, 
            const Tensor<T>& grad_bias,
            int batch_size = 1
        ) = 0;
};

#endif