#ifndef MAE_LOSS_HPP
#define MAE_LOSS_HPP

#include <cmath>
#include "../abstract/loss_function.hpp"

template <Numeric T> class MAELoss: public LossFunction<T> {
    public:
        T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const override;
        Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const override;

        friend std::ostream& operator<<(std::ostream& os, const MAELoss<T>& loss) {
            os << "MAELoss()";
            return os;
        }
};

#include "../../src/loss/mae_loss.tpp"

#endif