#ifndef AVG_POOLING_TPP
#define AVG_POOLING_TPP

#include "../../inc/layer/avg_pooling.hpp"

template <Numeric T>
AvgPooling<T>::AvgPooling(int kernel_size, int stride, int padding) {
    if(kernel_size <= 0 || stride <= 0) {
        throw std::invalid_argument("Kernel size and stride must be positive integers");
    }

    if(stride == -1) {
        stride = kernel_size;
    } else {
        this->stride = stride;
    }

    this->kernel_size = kernel_size;
    this->padding = padding;
}

template <Numeric T>
Tensor<T> AvgPooling<T>::forward(const Tensor<T>& input) {
    this->input_shape = input.shape();
    int channels = input.shape(0);
    int height = input.shape(1);
    int width = input.shape(2);

    int output_height = (height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;

    Shape output_shape{channels, output_height, output_width};
    output = Tensor<T>(output_shape);

    #pragma omp parallel for collapse(3)
    for(int c = 0; c < channels; ++c) {
        for(int h = 0; h < output_height; ++h) {
            for(int w = 0; w < output_width; ++w) {
                int h_start = std::max(0, h * stride - padding);
                int h_end = std::min(h_start + kernel_size, height);
                int w_start = std::max(0, w * stride - padding);
                int w_end = std::min(w_start + kernel_size, width);

                T total = 0;
                for(int i = h_start; i < h_end; ++i) {
                    for(int j = w_start; j < w_end; ++j) {
                        total += input(c, i, j).value();
                    }
                }

                output(c, h, w) = total / (kernel_size * kernel_size);
            }
        }
    }

    return output;
}

template <Numeric T>
Tensor<T> AvgPooling<T>::backward(const Tensor<T>& grad_output) {
    if(grad_output.shape() != output.shape()) {
        throw std::invalid_argument("Gradient shape doesn't match layer's output shape");
    }

    Tensor<T> grad_input(input_shape);

    #pragma omp parallel for collapse(3)
    for(int c = 0; c < grad_output.shape(0); ++c) {
        for(int h = 0; h < grad_output.shape(1); ++h) {
            for(int w = 0; w < grad_output.shape(2); ++w) {
                int h_start = std::max(0, h * stride - padding);
                int h_end = std::min(h_start + kernel_size, input_shape[1]);
                int w_start = std::max(0, w * stride - padding);
                int w_end = std::min(w_start + kernel_size, input_shape[2]);

                T grad_val = grad_output(c, h, w).value() / (kernel_size * kernel_size);
                for(int i = h_start; i < h_end; ++i) {
                    for(int j = w_start; j < w_end; ++j) {
                        grad_input(c, i, j) += grad_val;
                    }
                }
            }
        }
    }

    return grad_input;
}

#endif