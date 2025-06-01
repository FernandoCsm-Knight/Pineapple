#ifndef ADAM_HPP
#define ADAM_HPP

#include <cmath>

#include "../abstract/optimizer.hpp"

template <Numeric T> class Adam: public Optimizer<T> {
    private:
        T beta1;
        T beta2;
        T epsilon;
        int t;
        
        Tensor<T>* weight_m;
        Tensor<T>* weight_v;
        Tensor<T>* bias_m;
        Tensor<T>* bias_v;
    
    public:
        Adam(T learning_rate, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);
        ~Adam();

        void step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) override;

        friend std::ostream& operator<<(std::ostream& os, const Adam<T>& adam) {
            os << "Adam(learning_rate=" << adam.learning_rate << ", beta1=" << adam.beta1 << ", beta2=" << adam.beta2 << ", epsilon=" << adam.epsilon << ")";
            return os;
        }
};

#include "../../src/optimizer/adam.tpp"
#endif