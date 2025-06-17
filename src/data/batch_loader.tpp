#ifndef BATCH_LOADER_TPP
#define BATCH_LOADER_TPP

#include "../../inc/data/batch_loader.hpp"

template <Numeric T>
BatchLoader<T>::BatchLoader(const Tensor<T>& data, const Tensor<T>& target, int batch_size, bool shuffle, int seed)
    : data(data), target(target), size(batch_size), samples_count(data.shape(0)), seed(seed), shuffle(shuffle) {
    if(data.shape(0) != target.shape(0)) {
        throw std::invalid_argument("Data and target must have the same number of samples");
    }
    
    gen.seed(seed);

    indices = new int[samples_count];
    #pragma omp parallel for if(samples_count > 1000)
    for(int i = 0; i < samples_count; ++i) {
        indices[i] = i;
    }

    if(shuffle) std::shuffle(indices, indices + samples_count, gen);
}

template <Numeric T>
BatchLoader<T>::~BatchLoader() {
    delete[] indices;
}

template <Numeric T>
bool BatchLoader<T>::has_next() const {
    return current_index < samples_count;
}

template <Numeric T>
int BatchLoader<T>::num_batches() const {
    return (samples_count + size - 1) / size;
}

template <Numeric T>
int BatchLoader<T>::current_batch() const {
    return current_index / size;
}

template <Numeric T>
int BatchLoader<T>::batch_size() const {
    return size;
}

template <Numeric T>
int BatchLoader<T>::num_samples() const {
    return samples_count;
}

template <Numeric T>
std::pair<Tensor<T>, Tensor<T>> BatchLoader<T>::next() {
    if(!has_next()) {
        throw std::out_of_range("No more batches available");
    }

    int start = current_index;
    int end = std::min(start + size, samples_count);
    int curr_size = end - start;

    Tensor<T> batch_data(curr_size, data.shape(1));
    Tensor<T> batch_target(curr_size);

    for(int i = 0; i < curr_size; ++i) {
        const int idx = indices[start + i];

        batch_data(i) = data(idx);
        batch_target[i] = target[idx];
    }

    current_index += size;
    
    return std::make_pair(batch_data, batch_target);
}

template <Numeric T>
BatchLoader<T>::BatchIterator::BatchIterator(BatchLoader<T>* loader): loader(loader) {
    if(loader && !loader->has_next()) {
        loader = nullptr;
    }
}

template <Numeric T>
std::pair<Tensor<T>, Tensor<T>> BatchLoader<T>::BatchIterator::operator*() {
    return loader->next();
}

template <Numeric T>
typename BatchLoader<T>::BatchIterator& BatchLoader<T>::BatchIterator::operator++() {
    if(loader && !loader->has_next()) {
        loader = nullptr;
    } 

    return *this;
}

template <Numeric T>
bool BatchLoader<T>::BatchIterator::operator==(const BatchIterator& other) const {
    return loader == other.loader;
}

template <Numeric T>
bool BatchLoader<T>::BatchIterator::operator!=(const BatchIterator& other) const {
    return loader != other.loader;
}

template <Numeric T>
typename BatchLoader<T>::BatchIterator BatchLoader<T>::begin() {
    current_index = 0;
    if(shuffle) std::shuffle(indices, indices + samples_count, gen);
    return BatchIterator(this);
}

template <Numeric T>
typename BatchLoader<T>::BatchIterator BatchLoader<T>::end() {
    return BatchIterator(nullptr);
}

#endif