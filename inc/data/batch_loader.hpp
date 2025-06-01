#ifndef BATCH_LOADER_HPP
#define BATCH_LOADER_HPP

#include <random>

#include "../tensor/tensor.hpp"

template <Numeric T> class BatchLoader {
    private:
        Tensor<T> data;
        Tensor<T> target;

        int size;
        int samples_count;
        int current_index = 0;

        int* indices = nullptr;

        int seed;
        bool shuffle;
        std::mt19937 gen;

    public:
        BatchLoader(const Tensor<T>& data, const Tensor<T>& target, int batch_size, bool shuffle = true, int seed = 42);

        ~BatchLoader();

        std::pair<Tensor<T>, Tensor<T>> next();

        bool has_next() const;
        int num_batches() const;
        int current_batch() const;
        int batch_size() const;
        int num_samples() const;

        class BatchIterator {
            private:
                BatchLoader<T>* loader;

            public:
                BatchIterator(BatchLoader<T>* loader);

                std::pair<Tensor<T>, Tensor<T>> operator*();
                BatchIterator& operator++();
                bool operator==(const BatchIterator& other) const;
                bool operator!=(const BatchIterator& other) const;
        };

        BatchIterator begin();    
        BatchIterator end();
};

#include "../../src/data/batch_loader.tpp"

#endif