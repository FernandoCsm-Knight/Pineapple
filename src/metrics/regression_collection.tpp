#ifndef REGRESSION_COLLECTION_TPP
#define REGRESSION_COLLECTION_TPP

#include "../../inc/metrics/regression_collection.hpp"

template <Numeric T>
RegressionCollection<T>::RegressionCollection() {}

template <Numeric T>
RegressionCollection<T>::RegressionCollection(std::initializer_list<Metric<T>*> metrics) {
    for(Metric<T>* metric : metrics) {
        this->metrics[metric->name()] = metric;
    }
}

template <Numeric T>
void RegressionCollection<T>::update(const Tensor<T>& predictions, const Tensor<T>& targets) {
    if(predictions.length() != targets.length()) {
        throw std::invalid_argument("Predictions and targets must have the same length");
    }

    last_predictions = predictions;
    last_targets = targets;
}

template <Numeric T>
float RegressionCollection<T>::compute(const std::string& metric, int class_idx) const {
    if(!this->has_metric(metric)) {
        throw std::invalid_argument("Metric not found: " + metric);
    }

    if(last_predictions.length() == 0 || last_targets.length() == 0) {
        throw std::runtime_error("Empty cache. Call update() before compute().");
    }

    Metric<T>* m = this->metrics.at(metric);
    return m->compute(last_predictions, last_targets);
}

template <Numeric T>
void RegressionCollection<T>::reset() {
    last_predictions = Tensor<T>();
    last_targets = Tensor<T>();
}

#endif 