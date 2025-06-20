#ifndef METRIC_COLLECTION_HPP
#define METRIC_COLLECTION_HPP

#include "../abstract/metric.hpp"
#include "../tensor/tensor.hpp"

#include <initializer_list>
#include <unordered_map>
#include <set>

template <Numeric T> class MetricCollection {
    private:
        int** cm = nullptr;
        int num_classes = 0;
        std::unordered_map<std::string, Metric*> metrics;

        inline float average_micro(Metric* metric) const;
        inline float average_macro(Metric* metric) const;
        inline float average_weighted(Metric* metric) const;

    public:
        MetricCollection();
        MetricCollection(int num_classes);
        MetricCollection(int num_classes, std::initializer_list<Metric*> metrics);

        ~MetricCollection();

        void init_matrix(int num_classes);

        inline int classes() const;
        inline Tensor<int> confusion_matrix() const;

        inline float true_positive(int class_idx) const;
        inline float true_negative(int class_idx) const;
        inline float false_positive(int class_idx) const;
        inline float false_negative(int class_idx) const;

        bool has_metric(const std::string& metric) const;
        bool add_metric(Metric* metric);

        std::set<std::string> all_metrics() const;

        void update(const Tensor<T>& predictions, const Tensor<T>& targets);
        float compute(const std::string& metric, int class_idx = -1) const;
        
        void reset();
};

#include "../../src/metrics/metric_collection.tpp"

#endif 