#ifndef RECALL_HPP
#define RECALL_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class Recall: public Metric<T> {
    public:
        Recall(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/recall.tpp"

#endif