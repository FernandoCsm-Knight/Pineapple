#ifndef ACCURACY_TPP
#define ACCURACY_TPP

#include "../../inc/metrics/accuracy.hpp"

Accuracy::Accuracy(Average avg): Metric(avg) {}

std::string Accuracy::name() const {
    return "accuracy";
}

float Accuracy::compute(int TP, int TN, int FP, int FN) const {
    const float total = TP + TN + FP + FN;
    return (total == 0) ? 0 : static_cast<float>(TP + TN) / total;
}

#endif