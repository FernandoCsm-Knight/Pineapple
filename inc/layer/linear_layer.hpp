#ifndef LINEARLAYER_HPP
#define LINEARLAYER_HPP

#include "../types/numeric.hpp"
#include "../tensor/tensor.hpp"

#include "../abstract/layer.hpp"
#include "../abstract/optimizer.hpp"

template <Numeric T> class LinearLayer: public Layer<T> {
    private:
        int in_features;
        int out_features;

        Optimizer<T>* optimizer;

        Tensor<T> last_input;
        Tensor<T> weights;
        Tensor<T> bias;

    public:
        LinearLayer(int in_features, int out_features, Optimizer<T>* optim);
        ~LinearLayer();

        Tensor<T> forward(const Tensor<T>& input) override;

        Tensor<T> backward(const Tensor<T>& grad_weights) override;

        friend std::ostream& operator<<(std::ostream& os, const LinearLayer<T>& layer) {
            os << "LinearLayer(in_features=" << layer.in_features << ", out_features=" << layer.out_features;
            os << ", optimizer=" << *layer.optimizer << ")";
            return os;
        }

};

#include "../../src/layer/linear_layer.tpp"

#endif
