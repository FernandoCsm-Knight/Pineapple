#ifndef DROPOUT_CUDA_WRAPPERS_HPP
#define DROPOUT_CUDA_WRAPPERS_HPP

#include "cuda_macros.hpp"
#include <stdexcept>
#include <iostream>

namespace cuda_dropout_ops {

template<typename T>
void launch_dropout_mask(bool* mask, T dropout_rate, size_t size);

template<typename T>
void launch_dropout_forward(const T* input, const bool* mask, T* output, T scale, size_t size);

template<typename T>
void launch_dropout_backward(const T* grad_output, const bool* mask, T* grad_input, T scale, size_t size);

}

#endif
