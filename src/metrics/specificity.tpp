#ifndef SPECIFICITY_TPP
#define SPECIFICITY_TPP

#include "../../inc/metrics/specificity.hpp"

template <Numeric T>
Specificity<T>::Specificity(Average avg): Metric<T>(avg) {}

template <Numeric T>
std::string Specificity<T>::name() const {
    return "specificity";
}

template <Numeric T>
float Specificity<T>::compute(int TP, int TN, int FP, int FN) const {
    return (TN + FP == 0) ? 0.0f : static_cast<float>(TN) / (TN + FP);
}

template <Numeric T>
float Specificity<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    throw std::runtime_error("Specificity metric requires integer predictions and targets.");
}

#endif