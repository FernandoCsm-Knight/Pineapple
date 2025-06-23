#ifndef ACCURACY_HPP
#define ACCURACY_HPP

#include  "../abstract/metric.hpp"

template <Numeric T> class Accuracy: public Metric<T> {
    public:
        Accuracy(Average avg = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/accuracy.tpp"

#endif