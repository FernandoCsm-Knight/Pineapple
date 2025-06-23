#ifndef ACCURACY_TPP
#define ACCURACY_TPP

#include "../../inc/metrics/accuracy.hpp"

template <Numeric T>
Accuracy<T>::Accuracy(Average avg): Metric<T>(avg) {}

template <Numeric T>
std::string Accuracy<T>::name() const {
    return "accuracy";
}

template <Numeric T>
float Accuracy<T>::compute(int TP, int TN, int FP, int FN) const {
    const float total = TP + TN + FP + FN;
    return (total == 0) ? 0 : static_cast<float>(TP + TN) / total;
}

template <Numeric T>
float Accuracy<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    throw std::runtime_error("Accuracy metric requires integer predictions and targets.");
}

#endif