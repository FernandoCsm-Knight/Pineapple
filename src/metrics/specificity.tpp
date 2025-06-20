#ifndef SPECIFICITY_TPP
#define SPECIFICITY_TPP

#include "../../inc/metrics/specificity.hpp"

Specificity::Specificity(Average avg): Metric(avg) {}

std::string Specificity::name() const {
    return "specificity";
}

float Specificity::compute(int TP, int TN, int FP, int FN) const {
    return (TN + FP == 0) ? 0.0f : static_cast<float>(TN) / (TN + FP);
}

#endif