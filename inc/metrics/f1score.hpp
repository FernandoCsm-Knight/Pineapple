#ifndef F1_SCORE_HPP
#define F1_SCORE_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class F1Score: public Metric<T> {
    public:
        F1Score(Average average = Average::macro);

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/f1score.tpp"

#endif