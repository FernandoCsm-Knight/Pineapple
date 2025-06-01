#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

#include <random>

#include "../tensor/shape.hpp"
#include "../tensor/tensor.hpp"
#include "../types/numeric.hpp"
#include "../abstract/optimizer.hpp"

template <Numeric T> class ConvolutionalLayer {
    private:
        Shape input_shape;
        Shape output_shape;
        Shape kernels_shape;
        
        int stride;
        T learning_rate;
        
        Tensor<T> input;
        Tensor<T> kernels;
        Tensor<T> bias;
        
    public:
        ConvolutionalLayer(const Shape& input_shape, const Shape& kernel_shape, float lr = 1e-2, int stride = 1);    
        ~ConvolutionalLayer() = default;
        
        Tensor<T> forward(const Tensor<T>& input);
        Tensor<T> backward(const Tensor<T>& grad_output);

        void set_kernels(const Tensor<T>& new_kernels);
        void set_bias(const Tensor<T>& new_bias);

        friend std::ostream& operator<<(std::ostream& os, const ConvolutionalLayer<T>& layer) {
            os << "ConvolutionalLayer(input_shape=" << layer.input_shape << ", kernel_shape=" << layer.kernels_shape << ")";
            return os;
        }
};

#include "../../src/layer/convolutional_layer.tpp"

#endif