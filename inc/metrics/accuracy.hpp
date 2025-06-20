#ifndef ACCURACY_HPP
#define ACCURACY_HPP

#include  "../abstract/metric.hpp"

class Accuracy: public Metric {
    public:
        Accuracy(Average avg = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
};

#include "../../src/metrics/accuracy.tpp"

#endif