#ifndef PARTITION_HPP
#define PARTITION_HPP

#include <random>
#include <tuple>

#include "../tensor/tensor.hpp"

template <Numeric T> class Partition {
    private:
        Tensor<T> data;
        Tensor<T> target;

        bool shuffle;
        int* indices = nullptr;
        int samples_count = 0;

        int seed = 42;
        std::mt19937 gen;

        std::tuple<Shape, Shape, Shape, Shape> split_shape(float train_ratio, float test_ratio, const Shape& data_shape, const Shape& target_shape);

    public:
        Partition(const Tensor<T>& data, const Tensor<T>& target, bool shuffle = false);
        ~Partition();

        void random_seed(int seed);

        std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> stratified_split(const Tensor<T>& stratify, float test_ratio);
        std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>> split(float test_ratio);
};

#include "../../src/data/partition.tpp"

#endif