#ifndef MIN_POOLING_TPP
#define MIN_POOLING_TPP

#include "../../inc/layer/min_pooling.hpp"

template <Numeric T>
MinPooling<T>::MinPooling(int kernel_size, int stride, int padding) {
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
Tensor<T> MinPooling<T>::forward(const Tensor<T>& input) {
    this->input = input;
    int channels = input.shape(0);
    int height = input.shape(1);
    int width = input.shape(2);

    int output_height = (height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;

    Shape output_shape{channels, output_height, output_width};
    min_indices = Tensor<T>(output_shape);
    output = Tensor<T>(output_shape);

    #pragma omp parallel for collapse(3)
    for(int c = 0; c < channels; ++c) {
        for(int h = 0; h < output_height; ++h) {
            for(int w = 0; w < output_width; ++w) {
                int h_start = std::max(0, h * stride - padding);
                int h_end = std::min(h_start + kernel_size, height);
                int w_start = std::max(0, w * stride - padding);
                int w_end = std::min(w_start + kernel_size, width);

                size_t min_idx = h_start * width + w_start;
                T min_val = input(c, h_start, w_start).value();
                for(int i = h_start; i < h_end; ++i) {
                    for(int j = w_start; j < w_end; ++j) {
                        T val = input(c, i, j).value();

                        if(val < min_val) {
                            min_val = val;
                            min_idx = i * width + j;
                        }
                    }
                }

                min_indices(c, h, w) = min_idx;
                output(c, h, w) = min_val;
            }
        }
    }

    return output;
}

template <Numeric T>
Tensor<T> MinPooling<T>::backward(const Tensor<T>& grad_output) {
    if(grad_output.shape() != output.shape()) {
        throw std::invalid_argument("Gradient shape doesn't match layer's output shape");
    }

    Tensor<T> grad_input(input_shape);

    #pragma omp parallel for collapse(3)
    for(int c = 0; c < grad_output.shape(0); ++c) {
        for(int h = 0; h < grad_output.shape(1); ++h) {
            for(int w = 0; w < grad_output.shape(2); ++w) {
                size_t min_idx = min_indices(c, h, w).value();
                grad_input[min_idx] += grad_output(c, h, w).value();
            }
        }
    }

    return grad_input;
}

#endif