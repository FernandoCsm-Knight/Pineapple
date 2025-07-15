#ifndef METRIC_COLLECTION_TPP
#define METRIC_COLLECTION_TPP

#include "../../inc/abstract/metric_collection.hpp"

template <Numeric T>
MetricCollection<T>::~MetricCollection() {
    for(auto& entry : metrics) {
        delete entry.second;
    }

    metrics.clear();
}

template <Numeric T>
Metric<T>* MetricCollection<T>::get(const std::string& metric) {
    if(!has_metric(metric)) {
        throw std::invalid_argument("Metric '" + metric + "' does not exist in the collection.");
    }
    
    return metrics[metric];
}

template <Numeric T>
bool MetricCollection<T>::has_metric(const std::string& metric) const {
    return metrics.find(metric) != metrics.end();
}

template <Numeric T>
bool MetricCollection<T>::add_metric(Metric<T>* metric) {
    bool response = !has_metric(metric->name());
    if(response) metrics[metric->name()] = metric;
    return response;
}

template <Numeric T>
std::set<std::string> MetricCollection<T>::all_metrics() const {
    std::set<std::string> names;

    for(const auto& entry : metrics) {
        names.insert(entry.first);
    }
    
    return names;
}

template <Numeric T>
inline int MetricCollection<T>::size() const {
    return metrics.size();
}

#endif 