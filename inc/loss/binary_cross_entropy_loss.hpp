#ifndef BINARY_CROSS_ENTROPY_LOSS_HPP
#define BINARY_CROSS_ENTROPY_LOSS_HPP

#include <cmath>
#include "../abstract/loss_function.hpp"

template <Numeric T> class BinaryCrossEntropyLoss: public LossFunction<T> {
    public:
        T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const override;
        Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const override;

        friend std::ostream& operator<<(std::ostream& os, const BinaryCrossEntropyLoss<T>& loss) {
            os << "BinaryCrossEntropyLoss()";
            return os;
        }
};

#include "../../src/loss/binary_cross_entropy_loss.tpp"

#endif