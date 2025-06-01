#ifndef MAX_POOLING_HPP
#define MAX_POOLING_HPP

#include "../tensor/tensor.hpp"
#include "../types/numeric.hpp"
#include "../abstract/layer.hpp"

template <Numeric T> class MaxPooling: public Layer<T> {
    private:
        int kernel_size;
        int stride;
        int padding;

        Shape input_shape;

        Tensor<T> output;
        Tensor<size_t> max_indices;

    public:
        MaxPooling(int kernel_size, int stride = -1, int padding = 0);
        ~MaxPooling() = default;

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        friend std::ostream& operator<<(std::ostream& os, const MaxPooling<T>& layer) {
            os << "MaxPooling(kernel_size=" << layer.kernel_size << ", stride=" << layer.stride << ", padding=" << layer.padding << ")";
            return os;
        }
};

#include "../../src/layer/max_pooling.tpp"

#endif