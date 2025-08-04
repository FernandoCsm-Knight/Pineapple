#ifndef METRIC_COLLECTION_HPP
#define METRIC_COLLECTION_HPP

#include "support_device.hpp"
#include "../tensor/tensor.hpp"
#include "metric.hpp"

#include <unordered_map>
#include <set>

template <Numeric T> class MetricCollection: public SupportDevice {
    protected:
        std::unordered_map<std::string, Metric<T>*> metrics;

    public:
        virtual ~MetricCollection();

        Metric<T>* get(const std::string& metric);

        bool has_metric(const std::string& metric) const;
        bool add_metric(Metric<T>* metric);

        std::set<std::string> all_metrics() const;
        inline int size() const;

        virtual void update(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;
        virtual float compute(const std::string& metric, int class_idx = -1) const = 0;
    
        virtual void reset() = 0;
};

#include "../../src/abstract/metric_collection.tpp"

#endif 