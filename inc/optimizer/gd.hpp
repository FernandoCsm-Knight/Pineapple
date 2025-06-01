#ifndef GD_HPP
#define GD_HPP

#include "../abstract/optimizer.hpp"

template <Numeric T> class GD: public Optimizer<T> {
    private:
        bool regression;
    
    public:
        GD(T learning_rate, bool regression = false);

        void step(Tensor<T>& weights, const Tensor<T>& grad_weights, Tensor<T>& bias, const Tensor<T>& grad_bias, int batch_size) override;

        friend std::ostream& operator<<(std::ostream& os, const GD<T>& gd) {
            os << "GD(learning_rate=" << gd.learning_rate << ")";
            return os;
        }
};

#include "../../src/optimizer/gd.tpp"

#endif