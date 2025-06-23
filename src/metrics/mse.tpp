#ifndef MSE_TPP
#define MSE_TPP

#include "../../inc/metrics/mse.hpp"

template <Numeric T>
MSE<T>::MSE() {}

template <Numeric T>
std::string MSE<T>::name() const {
    return "mse";
}

template <Numeric T>
float MSE<T>::compute(int TP, int TN, int FP, int FN) const {
    return static_cast<float>(FP + FN) / (TP + TN + FP + FN);
}

template <Numeric T>
float MSE<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    if(predictions.length() != targets.length()) {
        throw std::invalid_argument("Predictions and targets must have the same length.");
    }

    const Tensor<T> residual = targets - predictions.squeeze();
    return (residual * residual).mean();
}

#endif 