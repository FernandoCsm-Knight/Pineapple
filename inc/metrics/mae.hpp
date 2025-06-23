#ifndef MAE_HPP
#define MAE_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class MAE: public Metric<T> {
    public:
        MAE();

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/mae.tpp"

#endif