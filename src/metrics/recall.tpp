#ifndef RECALL_TPP
#define RECALL_TPP

#include "../../inc/metrics/recall.hpp"

template <Numeric T>
Recall<T>::Recall(Average avg): Metric<T>(avg) {}

template <Numeric T>
std::string Recall<T>::name() const {
    return "recall";
}

template <Numeric T>
float Recall<T>::compute(int TP, int TN, int FP, int FN) const {
    return (TP + FN == 0) ? 0.0f : static_cast<float>(TP) / (TP + FN);
}

template <Numeric T>
float Recall<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) {
    throw std::runtime_error("Recall metric requires integer predictions and targets.");
}

#endif