#ifndef AVG_POOLING_HPP
#define AVG_POOLING_HPP

#include "../tensor/tensor.hpp"
#include "../types/numeric.hpp"
#include "../abstract/layer.hpp"

template <Numeric T> class AvgPooling: public Layer<T> {
    private:
        int kernel_size;
        int stride;
        int padding;

        Shape input_shape;
        Tensor<T> output;

    public:
        AvgPooling(int kernel_size, int stride = -1, int padding = 0);
        ~AvgPooling() = default;

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        friend std::ostream& operator<<(std::ostream& os, const AvgPooling<T>& layer) {
            os << "AvgPooling(kernel_size=" << layer.kernel_size << ", stride=" << layer.stride << ", padding=" << layer.padding << ")";
            return os;
        }
};

#include "../../src/layer/avg_pooling.tpp"

#endif