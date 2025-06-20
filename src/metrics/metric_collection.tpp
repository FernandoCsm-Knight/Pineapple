#ifndef METRIC_COLLECTION_TPP
#define METRIC_COLLECTION_TPP

#include "../../inc/metrics/metric_collection.hpp"

// Private methods

template <Numeric T>
float MetricCollection<T>::average_micro(Metric* metric) const {
    int TP = 0, FP = 0, FN = 0, TN = 0;

    for(int i = 0; i < num_classes; ++i) {
        TP += cm[i][i];
        
        for(int j = 0; j < num_classes; ++j) {
            if(j != i) {
                FP += cm[j][i];
                FN += cm[i][j];
            }
        }

        TN += this->true_negative(i);
    }

    return metric->compute(TP, TN, FP, FN);
}


template <Numeric T>
float MetricCollection<T>::average_macro(Metric* metric) const {
    float sum = 0.0f;
    int valid = 0;

    for(int i = 0; i < num_classes; ++i) {
        const int TP = cm[i][i];
        
        int FP = 0, FN = 0;
        for(int j = 0; j < num_classes; ++j) {
            if(j != i) {
                FP += cm[j][i];
                FN += cm[i][j];
            }
        }
        
        const int TN = this->true_negative(i);
        const float result = metric->compute(TP, TN, FP, FN);
        
        if(std::isfinite(result)) {
            sum += result;
            valid++;
        }
    }

    return (valid == 0) ? 0.0f : sum / valid;
}

template <Numeric T>
float MetricCollection<T>::average_weighted(Metric* metric) const {
    float sum = 0.0f;
    int total = 0;

    for(int i = 0; i < num_classes; ++i) {
        for(int j = 0; j < num_classes; ++j) {
            total += cm[i][j];
        }
    }

    if(total != 0) {
        for(int i = 0; i < num_classes; ++i) {
            const int TP = cm[i][i];
            
            int FP = 0, FN = 0;
            for(int j = 0; j < num_classes; ++j) {
                if(j != i) {
                    FP += cm[j][i];
                    FN += cm[i][j];
                }
            }
            
            const int TN = this->true_negative(i);
            const float result = metric->compute(TP, TN, FP, FN);
            
            int support = 0;
            for(int j = 0; j < num_classes; ++j) {
                support += cm[i][j];
            }
            
            if(std::isfinite(result)) {
                sum += result * support;
            }
        }
    }

    return (total == 0) ? 0.0f : sum / total;
}

// Constructors

template <Numeric T>
MetricCollection<T>::MetricCollection() {}

template <Numeric T>
MetricCollection<T>::MetricCollection(int num_classes) {
    this->init_matrix(num_classes);
}

template <Numeric T>
MetricCollection<T>::MetricCollection(int num_classes, std::initializer_list<Metric*> metrics) {
    this->init_matrix(num_classes);
    
    for(Metric* metric : metrics) {
        this->metrics[metric->name()] = metric;
    }
}

// Destructor

template <Numeric T>
MetricCollection<T>::~MetricCollection() {
    if(cm != nullptr) {
        for(int i = 0; i < num_classes; ++i) {
            delete[] cm[i];
        }
        delete[] cm;
    }

    for(const auto& entry : metrics) {
        delete entry.second;
    }
}

// Public methods

template <Numeric T>
void MetricCollection<T>::init_matrix(int num_classes) {
    if(cm != nullptr) {
        for(int i = 0; i < this->num_classes; ++i) {
            delete[] cm[i];
        }

        delete[] cm;
    }

    this->num_classes = num_classes;
    cm = new int*[num_classes];
    for(int i = 0; i < num_classes; ++i) {
        cm[i] = new int[num_classes]();
    }
}

template <Numeric T>
inline int MetricCollection<T>::classes() const {
    return this->num_classes;
}

template <Numeric T>
inline Tensor<int> MetricCollection<T>::confusion_matrix() const {
    Tensor<int> result(num_classes, num_classes);

    for(int i = 0; i < num_classes; ++i) {
        for(int j = 0; j < num_classes; ++j) {
            result[i * num_classes + j] = cm[i][j];
        }
    }

    return result;
}

template <Numeric T>
void MetricCollection<T>::update(const Tensor<T>& predictions, const Tensor<T>& targets) {
    if(predictions.length() != targets.length()) {
        throw std::invalid_argument("Predictions and targets must have the same length.");
    }

    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    for(size_t i = 0; i < predictions.length(); ++i) {
        const int target = targets[i];
        const int prediction = predictions[i];

        cm[target][prediction]++;
    }
}

template <Numeric T>
void MetricCollection<T>::reset() {
    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    for(int i = 0; i < num_classes; ++i) {
        for(int j = 0; j < num_classes; ++j) {
            cm[i][j] = 0;
        }
    }
}

template <Numeric T>
inline float MetricCollection<T>::true_positive(int class_idx) const {
    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    return cm[class_idx][class_idx];
}

template <Numeric T>
inline float MetricCollection<T>::true_negative(int class_idx) const {
    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    float TN = 0.0f;

    for(int i = 0; i < num_classes; ++i) {
        for(int j = 0; j < num_classes; ++j) {
            if(i != class_idx && j != class_idx) {
                TN += cm[i][j];
            }
        }
    }

    return TN;
}

template <Numeric T>
inline float MetricCollection<T>::false_positive(int class_idx) const { 
    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    float FP = 0.0f;

    for(int i = 0; i < num_classes; ++i) {
        if(i != class_idx) {
            FP += cm[i][class_idx];
        }
    }

    return FP;
}

template <Numeric T>
inline float MetricCollection<T>::false_negative(int class_idx) const {
    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }
    
    float FN = 0.0f;

    for(int i = 0; i < num_classes; ++i) {
        if(i != class_idx) {
            FN += cm[class_idx][i];
        }
    }

    return FN;
}

template <Numeric T>
bool MetricCollection<T>::has_metric(const std::string& metric) const {
    return metrics.find(metric) != metrics.end();
}

template <Numeric T>
bool MetricCollection<T>::add_metric(Metric* metric) {
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
float MetricCollection<T>::compute(const std::string& metric, int class_idx) const {
    if(metrics.find(metric) == metrics.end()) {
        throw std::invalid_argument("Metric not found: " + metric);
    }

    if(cm == nullptr) {
        throw std::runtime_error("Confusion matrix not initialized. Call init_matrix() first.");
    }

    Metric* m = metrics.at(metric);
    float result = 0.0f;

    if(class_idx == -1) {
        switch(m->average_type()) {
            case Average::micro:
                result = average_micro(m);
                break;
            case Average::macro:
                result = average_macro(m);
                break;
            case Average::weighted:
                result = average_weighted(m);
                break;
            default:
                throw std::invalid_argument("Unknown average type.");
        }
    } else {
        const int TP = cm[class_idx][class_idx];

        int FP = 0, FN = 0;
        for(int i = 0; i < num_classes; ++i) {
            if(i != class_idx) {
                FP += cm[i][class_idx];
                FN += cm[class_idx][i];
            }
        }

        const int TN = this->true_negative(class_idx);
        result = m->compute(TP, TN, FP, FN);
    }

    return result;
}

#endif