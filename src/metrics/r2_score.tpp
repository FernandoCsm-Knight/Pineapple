#ifndef R2_SCORE_TPP
#define R2_SCORE_TPP

#include "../../inc/metrics/r2_score.hpp"

template <Numeric T>
R2Score<T>::R2Score() {}

template <Numeric T>
std::string R2Score<T>::name() const {
    return "r2";        
}

template <Numeric T>
float R2Score<T>::compute(int TP, int TN, int FP, int FN) const {
    throw std::logic_error("R2Score does not support confusion matrix inputs.");
}

template <Numeric T>
float R2Score<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    if(predictions.length() != targets.length()) {
        throw std::invalid_argument("Predictions and targets must have the same length.");
    }
    
    const T mean_targets = targets.mean();
    
    const Tensor<T> deviation = targets - mean_targets;
    const Tensor<T> ss_tot = (deviation * deviation).sum();

    const Tensor<T> residual = targets - predictions.squeeze();
    const Tensor<T> ss_res = (residual * residual).sum();
    
    return (ss_tot.length() == 0) ? 0.0f : 1.0f - (ss_res.value() / ss_tot.value());
}

#endif