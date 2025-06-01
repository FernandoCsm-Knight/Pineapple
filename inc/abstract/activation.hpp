#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "layer.hpp"
#include "../tensor/tensor.hpp"

template <Numeric T> class Activation: public Layer<T> {
    protected:
        Tensor<T> last_output;
        bool last_activation = false;

        virtual Tensor<T> apply(const Tensor<T>& input) const = 0;
        virtual Tensor<T> derivative(const Tensor<T>& input) const = 0;

    public:
        Activation() = default;

        virtual Tensor<T> forward(const Tensor<T>& input) override;
        virtual Tensor<T> backward(const Tensor<T>& grad_output) override;
        
        virtual bool is_activation() const override;
        virtual void set_last_activation(bool is_last);
        virtual bool is_last_activation() const;
};

#include "../../src/abstract/activation.tpp"

#endif 
