#ifndef HUBER_LOSS_HPP
#define HUBER_LOSS_HPP

#include <cmath>
#include "../abstract/loss_function.hpp"

template <Numeric T> class HuberLoss: public LossFunction<T> {
    private:
        T delta;
    
    public:
        explicit HuberLoss(T delta = 1.0) : delta(delta) {}
        
        T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const override;
        Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const override;

        friend std::ostream& operator<<(std::ostream& os, const HuberLoss<T>& loss) {
            os << "HuberLoss(delta=" << loss.delta << ")";
            return os;
        }
};

#include "../../src/loss/huber_loss.tpp"

#endif