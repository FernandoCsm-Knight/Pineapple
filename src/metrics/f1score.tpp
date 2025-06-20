#ifndef F1_SCORE_TPP
#define F1_SCORE_TPP

#include "../../inc/metrics/f1score.hpp"

F1Score::F1Score(Average avg): Metric(avg) {}

std::string F1Score::name() const {
    return "f1_score";
}

float F1Score::compute(int TP, int TN, int FP, int FN) const {
    const float precision = (TP + FP == 0) ? 0.0f : static_cast<float>(TP) / (TP + FP);
    const float recall = (TP + FN == 0) ? 0.0f : static_cast<float>(TP) / (TP + FN);
    
    return (precision + recall == 0) ? 0.0f : 2 * (precision * recall) / (precision + recall);
}

#endif