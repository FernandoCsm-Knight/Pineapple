#ifndef PRECISION_TPP
#define PRECISION_TPP

#include "../../inc/metrics/precision.hpp"

Precision::Precision(Average avg): Metric(avg) {}

std::string Precision::name() const {
    return "precision";
}

float Precision::compute(int TP, int TN, int FP, int FN) const {
    return (TP + FP == 0) ? 0.0f : static_cast<float>(TP) / (TP + FP);
}

#endif