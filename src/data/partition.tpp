#ifndef PARTITION_TPP
#define PARTITION_TPP

#include "../../inc/data/partition.hpp"

// Private Methods

template <Numeric T>
std::tuple<Shape, Shape, Shape, Shape> Partition<T>::split_shape(float train_ratio, float test_ratio, const Shape& data_shape, const Shape& target_shape) {
    const int train_count = static_cast<int>(train_ratio * samples_count);
    const int test_count = static_cast<int>(test_ratio * samples_count);

    Shape train_data_shape(train_count);
    Shape test_data_shape(test_count);
    Shape train_target_shape(train_count);
    Shape test_target_shape(test_count);

    for(int i = 1; i < data.ndim(); ++i) {
        train_data_shape.add_dimension(data_shape[i]);
        if(test_ratio > 0) test_data_shape.add_dimension(data_shape[i]);
    }

    for(int i = 1; i < target.ndim(); ++i) {
        train_target_shape.add_dimension(target_shape[i]);
        if(test_ratio > 0) test_target_shape.add_dimension(target_shape[i]);
    }

    return std::make_tuple(train_data_shape, train_target_shape, test_data_shape, test_target_shape);
}

// Public Methods

template <Numeric T>
Partition<T>::Partition(const Tensor<T>& data, const Tensor<T>& target, bool shuffle): data(data), target(target), shuffle(shuffle) {
    if(data.shape(0) != target.shape(0)) {
        throw std::invalid_argument("Data and target must have the same number of samples");
    }
    
    samples_count = data.shape(0);
    indices = new int[samples_count];
    
    #pragma omp parallel for if(samples_count > 1000)
    for(int i = 0; i < samples_count; ++i) {
        indices[i] = i;
    }
    
    gen.seed(seed);
}

template <Numeric T>
void Partition<T>::random_seed(int seed) {
    gen.seed(seed);
    this->seed = seed;
}

template <Numeric T>
Partition<T>::~Partition() {
    delete[] indices;
}

template <Numeric T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Partition<T>::stratified_split(const Tensor<T>& stratify, float test_ratio) {
    if(test_ratio > 1.0f) {
        throw std::invalid_argument("Test ratio must be 1 or less");
    }

    if(stratify.shape(0) != samples_count) {
        throw std::invalid_argument("Stratify tensor must have the same number of samples as data and target");
    }

    const float train_ratio = 1.0f - test_ratio;
    const auto [train_data_shape, train_target_shape, test_data_shape, test_target_shape] = split_shape(train_ratio, test_ratio, data.shape(), target.shape());

    Tensor<T> train_data(train_data_shape);
    Tensor<T> test_data(test_data_shape);
    Tensor<T> train_target(train_target_shape);
    Tensor<T> test_target(test_target_shape);

    std::unordered_map<T, std::vector<int>> stratified_indices;
    for(int i = 0; i < samples_count; ++i) {
        stratified_indices[stratify[i]].push_back(i);
    }

    int train_index = 0;
    int test_index = 0;

    for(size_t key = 0; key < stratified_indices.size(); ++key) {
        const std::vector<int>& indices = stratified_indices.at(key);

        std::vector<int> shuffled_indices = indices;
        if(shuffle) std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), gen);

        const int train_slice = static_cast<int>(train_ratio * indices.size());
        const int test_slice = static_cast<int>(test_ratio * indices.size());

        for(int i = 0; i < train_slice; ++i) {
            const int idx = shuffled_indices[i];
            
            train_data(train_index) = data(idx);
            train_target[train_index++] = target[idx];
        }

        for(int i = 0; i < test_slice; ++i) {
            const int idx = shuffled_indices[train_slice + i];
            
            test_data(test_index) = data(idx);
            test_target[test_index++] = target[idx];
        }
    }

    return std::make_tuple(train_data, train_target, test_data, test_target);
}

template <Numeric T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> Partition<T>::split(float test_ratio) {
    if(test_ratio > 1.0f) {
        throw std::invalid_argument("Test ratio must be 1 or less");
    }

    const float train_ratio = 1.0f - test_ratio;
    const auto [train_data_shape, train_target_shape, test_data_shape, test_target_shape] = split_shape(train_ratio, test_ratio, data.shape(), target.shape());

    Tensor<T> train_data(train_data_shape);
    Tensor<T> test_data(test_data_shape);
    Tensor<T> train_target(train_target_shape);
    Tensor<T> test_target(test_target_shape);

    if(shuffle) std::shuffle(indices, indices + samples_count, gen);

    const int train_count = static_cast<int>(train_ratio * samples_count);
    const int test_count = static_cast<int>(test_ratio * samples_count);

    for(int i = 0; i < train_count; ++i) {
        const int idx = indices[i];

        train_data(i) = data(idx);
        train_target[i] = target[idx];
    }

    for(int i = 0; i < test_count; ++i) {
        const int idx = indices[train_count + i];

        test_data(i) = data(idx);
        test_target[i] = target[idx];
    }

    return std::make_tuple(train_data, train_target, test_data, test_target);
}

#endif