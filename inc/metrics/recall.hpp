#ifndef RECALL_HPP
#define RECALL_HPP

#include "../abstract/metric.hpp"

class Recall: public Metric {
    public:
        Recall(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
};

#include "../../src/metrics/recall.tpp"

#endif