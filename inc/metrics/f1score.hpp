#ifndef F1_SCORE_HPP
#define F1_SCORE_HPP

#include "../abstract/metric.hpp"

class F1Score: public Metric {
    public:
        F1Score(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
};

#include "../../src/metrics/f1score.tpp"

#endif