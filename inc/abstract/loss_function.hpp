#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include "../tensor/tensor.hpp"

template <Numeric T> class LossFunction {
    public:
        virtual ~LossFunction() = default;
        
        virtual void to(Device target_device) {};
        virtual T compute(const Tensor<T>& predictions, const Tensor<T>& targets) const = 0;
        virtual Tensor<T> gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const = 0;
};

#endif