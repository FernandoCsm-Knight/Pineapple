#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "../tensor/tensor.hpp"
#include "../abstract/layer.hpp"

template <Numeric T> class FlattenLayer: public Layer<T> {
    private:
        Shape input_shape;

    public:
        FlattenLayer();
        ~FlattenLayer() = default;

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        friend std::ostream& operator<<(std::ostream& os, const FlattenLayer<T>& layer) {
            os << "FlattenLayer()";
            return os;
        }
};

#include "../../src/layer/flatten_layer.tpp"

#endif 