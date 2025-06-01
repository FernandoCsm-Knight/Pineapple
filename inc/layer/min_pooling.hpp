#ifndef MIN_POOLING_HPP
#define MIN_POOLING_HPP


#include "../tensor/tensor.hpp"
#include "../types/numeric.hpp"
#include "../abstract/layer.hpp"

template <Numeric T> class MinPooling: public Layer<T> {
    private:
        int kernel_size;
        int stride;
        int padding;

        Shape input_shape;
        
        Tensor<T> output;
        Tensor<size_t> min_indices;

    public:
        MinPooling(int kernel_size, int stride = -1, int padding = 0);
        ~MinPooling() = default;

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        friend std::ostream& operator<<(std::ostream& os, const MinPooling<T>& layer) {
            os << "MinPooling(kernel_size=" << layer.kernel_size << ", stride=" << layer.stride << ", padding=" << layer.padding << ")";
            return os;
        }
};

#include "../../src/layer/min_pooling.tpp"

#endif