#ifndef ADAMW_HPP
#define ADAMW_HPP

#include <cmath>

#include "../abstract/optimizer.hpp"

template <Numeric T> class AdamW: public Optimizer<T> {
    private:
        T weight_decay;
        T beta1;
        T beta2;
        T epsilon;
        int t;
        
        Tensor<T>* weight_m;
        Tensor<T>* weight_v;
        Tensor<T>* bias_m;
        Tensor<T>* bias_v;
    
    public:
        AdamW(T learning_rate, T weight_decay = 0.01, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);
        ~AdamW();

        void step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) override;

        friend std::ostream& operator<<(std::ostream& os, const AdamW<T>& adamw) {
            os << "AdamW(learning_rate=" << adamw.learning_rate << ", weight_decay=" << adamw.weight_decay;
            os << ", beta1=" << adamw.beta1 << ", beta2=" << adamw.beta2 << ", epsilon=" << adamw.epsilon << ")";
            return os;
        }
};

#include "../../src/optimizer/adamw.tpp"
#endif