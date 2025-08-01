#ifndef ACTIVATION_CUDA_WRAPPERS_HPP
#define ACTIVATION_CUDA_WRAPPERS_HPP

#include "cuda_macros.hpp"
#include <stdexcept>
#include <iostream>

namespace cuda_activation_ops {

template<typename T>
void launch_relu_apply(const T* input, T* output, size_t size);

template<typename T>
void launch_relu_derivative(const T* input, T* output, size_t size);

template<typename T>
void launch_sigmoid_apply(const T* input, T* output, size_t size);

template<typename T>
void launch_sigmoid_derivative(const T* input, T* output, size_t size);

template<typename T>
void launch_tanh_apply(const T* input, T* output, size_t size);

template<typename T>
void launch_tanh_derivative(const T* input, T* output, size_t size);

template<typename T>
void launch_elu_apply(const T* input, T* output, T alpha, size_t size);

template<typename T>
void launch_elu_derivative(const T* input, T* output, T alpha, size_t size);

template<typename T>
void launch_leaky_relu_apply(const T* input, T* output, T alpha, size_t size);

template<typename T>
void launch_leaky_relu_derivative(const T* input, T* output, T alpha, size_t size);

template<typename T>
void launch_softmax_apply(const T* input, T* output, int batch_size, int num_classes, size_t size);

template<typename T>
void launch_softmax_derivative(const T* input, T* output, size_t size);

}

#endif
