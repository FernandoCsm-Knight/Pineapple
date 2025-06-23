#ifndef PRECISION_TPP
#define PRECISION_TPP

#include "../../inc/metrics/precision.hpp"

template <Numeric T>
Precision<T>::Precision(Average avg): Metric<T>(avg) {}

template <Numeric T>
std::string Precision<T>::name() const {
    return "precision";
}

template <Numeric T>
float Precision<T>::compute(int TP, int TN, int FP, int FN) const {
    return (TP + FP == 0) ? 0.0f : static_cast<float>(TP) / (TP + FP);
}

template <Numeric T>
float Precision<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    throw std::runtime_error("Precision metric requires integer predictions and targets.");
}

#endif