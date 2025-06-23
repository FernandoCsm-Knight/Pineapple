#ifndef MAE_TPP
#define MAE_TPP

#include "../../inc/metrics/mae.hpp"

template <Numeric T>
MAE<T>::MAE() {}

template <Numeric T>
std::string MAE<T>::name() const {
    return "mae";
}

template <Numeric T>
float MAE<T>::compute(int TP, int TN, int FP, int FN) const {
    return static_cast<float>(FP + FN) / (TP + TN + FP + FN);
}

template <Numeric T>
float MAE<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    if(predictions.length() != targets.length()) {
        throw std::invalid_argument("Predictions and targets must have the same shape.");
    }

    return (targets - predictions.squeeze()).abs().mean();
}

#endif