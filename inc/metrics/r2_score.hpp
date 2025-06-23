#ifndef R2_SCORE_HPP
#define R2_SCORE_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class R2Score: public Metric<T> {
    public:
        R2Score();

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/r2_score.tpp"

#endif