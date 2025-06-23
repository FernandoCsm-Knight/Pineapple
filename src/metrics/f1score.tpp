#ifndef F1_SCORE_TPP
#define F1_SCORE_TPP

#include "../../inc/metrics/f1score.hpp"

template <Numeric T>
F1Score<T>::F1Score(Average avg): Metric<T>(avg) {}

template <Numeric T>
std::string F1Score<T>::name() const {
    return "f1_score";
}

template <Numeric T>
float F1Score<T>::compute(int TP, int TN, int FP, int FN) const {
    const float precision = (TP + FP == 0) ? 0.0f : static_cast<float>(TP) / (TP + FP);
    const float recall = (TP + FN == 0) ? 0.0f : static_cast<float>(TP) / (TP + FN);
    
    return (precision + recall == 0) ? 0.0f : 2 * (precision * recall) / (precision + recall);
}

template <Numeric T>
float F1Score<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    throw std::runtime_error("F1Score metric requires integer predictions and targets.");
}

#endif