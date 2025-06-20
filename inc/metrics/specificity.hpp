#ifndef SPECIFICITY_HPP
#define SPECIFICITY_HPP

#include "../abstract/metric.hpp"

class Specificity: public Metric {
    public:
        Specificity(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
};

#include "../../src/metrics/specificity.tpp"

#endif 