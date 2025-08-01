#ifndef CONFUSION_MATRIX_COLLECTION_HPP
#define CONFUSION_MATRIX_COLLECTION_HPP

#include "../abstract/metric_collection.hpp"

#include <initializer_list>

template <Numeric T> class ConfusionMatrixCollection: public MetricCollection<T> {
    private:
        int** cm = nullptr;
        int num_classes = 0;

        inline float average_micro(Metric<T>* metric) const;
        inline float average_macro(Metric<T>* metric) const;
        inline float average_weighted(Metric<T>* metric) const;

    public:
        ConfusionMatrixCollection();
        ConfusionMatrixCollection(int num_classes);
        ConfusionMatrixCollection(int num_classes, std::initializer_list<Metric<T>*> metrics);

        ~ConfusionMatrixCollection() override;

        void to(Device target_device) override;
        void init_matrix(int num_classes);

        inline int classes() const;
        inline Tensor<int> confusion_matrix() const;

        inline float true_positive(int class_idx) const;
        inline float true_negative(int class_idx) const;
        inline float false_positive(int class_idx) const;
        inline float false_negative(int class_idx) const;

        void update(const Tensor<T>& predictions, const Tensor<T>& targets) override;
        float compute(const std::string& metric, int class_idx = -1) const override;
        
        void reset() override;
};

#include "../../src/metrics/confusion_matrix_collection.tpp"

#endif 