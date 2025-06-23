#ifndef REGRESSION_COLLECTION_HPP
#define REGRESSION_COLLECTION_HPP

#include "../abstract/metric_collection.hpp"

#include <initializer_list>

template <Numeric T> class RegressionCollection: public MetricCollection<T> {
    protected:
        Tensor<T> last_predictions;
        Tensor<T> last_targets;

    public:
        RegressionCollection();
        RegressionCollection(std::initializer_list<Metric<T>*> metrics);

        ~RegressionCollection() override = default;

        void update(const Tensor<T>& predictions, const Tensor<T>& targets) override;
        float compute(const std::string& metric, int class_idx = -1) const override;
        
        void reset() override;
};

#include "../../src/metrics/regression_collection.tpp"

#endif 