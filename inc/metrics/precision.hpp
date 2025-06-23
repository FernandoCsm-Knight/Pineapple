#ifndef PRECISION_HPP
#define PRECISION_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class Precision: public Metric<T> {
    public:
        Precision(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/precision.tpp"

#endif