#ifndef MSE_LOSS_HPP
#define MSE_LOSS_HPP

#include <cmath>
#include "../abstract/loss_function.hpp"

template <Numeric T> class MSELoss: public LossFunction<T> {
    private:
        Tensor<T> last_input;

    public:
        T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const override;
        Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const override;

        friend std::ostream& operator<<(std::ostream& os, const MSELoss<T>& loss) {
            os << "MSELoss()";
            return os;
        }
};

#include "../../src/loss/mse_loss.tpp"

#endif