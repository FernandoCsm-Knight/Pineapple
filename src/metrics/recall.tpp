#ifndef RECALL_TPP
#define RECALL_TPP

#include "../../inc/metrics/recall.hpp"

Recall::Recall(Average avg): Metric(avg) {}

std::string Recall::name() const {
    return "recall";
}

float Recall::compute(int TP, int TN, int FP, int FN) const {
    return (TP + FN == 0) ? 0.0f : static_cast<float>(TP) / (TP + FN);
}

#endif