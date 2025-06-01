#ifndef CONVOLUTIONAL_LAYER_TPP
#define CONVOLUTIONAL_LAYER_TPP

#include "../../inc/layer/convolutional_layer.hpp"

template <Numeric T>
ConvolutionalLayer<T>::ConvolutionalLayer(const Shape& input_shape, const Shape& kernel_shape, float lr, int stride) 
    : input_shape(input_shape), stride(stride), learning_rate(lr) {
    
    if (input_shape.ndim() != 3) {
        throw std::invalid_argument("Input shape must be a 3D tuple");
    }

    if (kernel_shape.ndim() != 3) {
        throw std::invalid_argument("Kernel shape must be a 3D tuple");
    }
    
    int input_depth = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int kernel_depth = kernel_shape[0];
    int kernel_height = kernel_shape[1];
    int kernel_width = kernel_shape[2];
    
    output_shape = Shape{
        kernel_depth,
        (input_height - kernel_height) / stride + 1,
        (input_width - kernel_width) / stride + 1
    };
    
    kernels_shape = Shape{kernel_depth, input_depth, kernel_height, kernel_width};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(2.0 / (input_depth * kernel_height * kernel_width)));
    
    kernels = Tensor<T>(kernels_shape);
    
    #pragma omp parallel for
    for (size_t i = 0; i < kernels.length(); ++i) {
        kernels[i] = dist(gen);
    }
    
    bias = Tensor<T>(output_shape);
    std::normal_distribution<T> bias_dist(0, 0.01);
    
    #pragma omp parallel for
    for (size_t i = 0; i < bias.length(); ++i) {
        bias[i] = bias_dist(gen);
    }
}

template <Numeric T>
Tensor<T> ConvolutionalLayer<T>::forward(const Tensor<T>& input) {
    if (input.shape() != input_shape) {
        throw std::invalid_argument("Input shape doesn't match layer's expected input shape");
    }

    this->input = input;
    Tensor<T> output = bias;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < kernels_shape[0]; ++i) {
        for (int j = 0; j < kernels_shape[1]; ++j) {
            Tensor<T> input_slice = input(j);
            Tensor<T> kernel_slice = kernels(i, j);
            
            output(i) += input_slice.cross_correlation(kernel_slice, stride);
        }
    }

    return output;
}

template <Numeric T>
Tensor<T> ConvolutionalLayer<T>::backward(const Tensor<T>& grad_output) {
    if(grad_output.shape() != output_shape) {
        throw std::invalid_argument("Gradient shape doesn't match layer's output shape");
    }

    Tensor<T> grad_input(input_shape);
    Tensor<T> grad_kernels(kernels_shape);
    
    if(stride > 1) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < kernels_shape[0]; ++i) {
            for (int j = 0; j < kernels_shape[1]; ++j) {
                Tensor<T> grad_output_dilated = grad_output(i).dilate(stride - 1);
                
                Tensor<T> grad_input_j = grad_input(j);
                Tensor<T> convolved_grad = grad_output_dilated.convolve(kernels(i, j), 1, Correlation::full);

                if(convolved_grad.shape() != grad_input_j.shape()) {
                    Shape target_shape = grad_input_j.shape();
                    Shape current_shape = convolved_grad.shape();
                    
                    if(current_shape[0] < target_shape[0] || current_shape[1] < target_shape[1]) {
                        if(current_shape[0] < target_shape[0]) {
                            Tensor<T> zeros(target_shape[0] - current_shape[0], current_shape[1]);
                            convolved_grad.append(zeros, 0);
                        }
                        
                        if(current_shape[1] < target_shape[1]) {
                            Tensor<T> zeros(convolved_grad.shape(0), target_shape[1] - convolved_grad.shape(1));
                            convolved_grad.append(zeros, 1);
                        }
                    }
                }
                
                grad_input_j += convolved_grad;
                grad_kernels(i, j) = input(j).cross_correlation(grad_output_dilated, 1);
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < kernels_shape[0]; ++i) {
            for (int j = 0; j < kernels_shape[1]; ++j) {
                Tensor<T> grad_output_slice = grad_output(i);
                grad_input(j) += grad_output_slice.convolve(kernels(i, j), stride, Correlation::full);
                grad_kernels(i, j) = input(j).cross_correlation(grad_output_slice, stride);
            }
        }
    }
    
    kernels -= grad_kernels * learning_rate;
    bias -= grad_output * learning_rate;
    
    return grad_input;
}

template <Numeric T>
void ConvolutionalLayer<T>::set_kernels(const Tensor<T>& new_kernels) {
    if (new_kernels.shape() != kernels_shape) {
        throw std::invalid_argument("New kernels shape doesn't match layer's expected kernels shape");
    }
    
    kernels = new_kernels;
}

template <Numeric T>
void ConvolutionalLayer<T>::set_bias(const Tensor<T>& new_bias) {
    if (new_bias.shape() != output_shape) {
        throw std::invalid_argument("New bias shape doesn't match layer's expected bias shape");
    }
    
    bias = new_bias;
}

#endif