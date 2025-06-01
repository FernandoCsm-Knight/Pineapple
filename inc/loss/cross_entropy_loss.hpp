#ifndef CROSS_ENTROPY_LOSS_HPP
#define CROSS_ENTROPY_LOSS_HPP

#include <cmath>

#include "../abstract/loss_function.hpp"

template <Numeric T> class CrossEntropyLoss: public LossFunction<T> {
    public:
        T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const override;
        Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const override;

        friend std::ostream& operator<<(std::ostream& os, const CrossEntropyLoss<T>& loss) {
            os << "CrossEntropyLoss()";
            return os;
        }
};

#include "../../src/loss/cross_entropy_loss.tpp"

#endif