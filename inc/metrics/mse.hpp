#ifndef MSE_HPP
#define MSE_HPP

#include "../abstract/metric.hpp"

template <Numeric T> class MSE: public Metric<T> { 
    public:
        MSE();

        std::string name() const override;
        float compute(int TP, int TN, int FP, int FN) const override;
        float compute(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../../src/metrics/mse.tpp"

#endif