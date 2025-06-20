#ifndef PRECISION_HPP
#define PRECISION_HPP

#include "../abstract/metric.hpp"

class Precision: public Metric {
    public:
        Precision(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
};

#include "../../src/metrics/precision.tpp"

#endif