#ifndef LAYER_HPP
#define LAYER_HPP

#include "../types/numeric.hpp"
#include "optimizer.hpp"

template <Numeric T> class Layer {
    protected:
        Optimizer<T>* optimizer = nullptr;
        bool training = true;

    public:
        virtual ~Layer();

        virtual Tensor<T> forward(const Tensor<T>& input) = 0;
        virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;

        virtual void train();
        virtual void eval();

        bool is_in_train_mode();

        virtual bool is_activation() const;
        virtual bool is_optimization() const;

        virtual void set_optimizer(Optimizer<T>* optim);
};

#include "../../src/abstract/layer.tpp"

#endif 