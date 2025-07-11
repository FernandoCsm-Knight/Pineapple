#ifndef SGD_HPP
#define SGD_HPP

#include "../abstract/optimizer.hpp"

template <Numeric T> class SGD: public Optimizer<T> {
    private:
        T momentum;
        Tensor<T>* weight_v;
        Tensor<T>* bias_v;

    public:
        SGD(T learning_rate, T momentum = 0.9f);
        ~SGD();

        Optimizer<T>* copy() const override;

        void step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias) override;

        friend std::ostream& operator<<(std::ostream& os, const SGD<T>& sgd) {
            os << "SGD(learning_rate=" << sgd.learning_rate << ", momentum=" << sgd.momentum << ")";
            return os;
        }

};

#include "../../src/optimizer/sgd.tpp"

#endif