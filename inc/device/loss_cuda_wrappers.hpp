#ifndef LOSS_CUDA_WRAPPERS_HPP
#define LOSS_CUDA_WRAPPERS_HPP

#include "cuda_macros.hpp"
#include <stdexcept>
#include <iostream>

namespace cuda_loss_ops {

// Binary Cross Entropy Loss
template<typename T>
T launch_binary_cross_entropy_compute(const T* predictions, const T* targets, size_t batch_size);

template<typename T>
void launch_binary_cross_entropy_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size);

// Cross Entropy Loss
template<typename T>
T launch_cross_entropy_compute(const T* predictions, const T* targets, size_t batch_size, size_t num_classes);

template<typename T>
void launch_cross_entropy_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size, size_t num_classes);

// MSE Loss
template<typename T>
T launch_mse_compute(const T* predictions, const T* targets, size_t batch_size);

template<typename T>
void launch_mse_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size);

// MAE Loss
template<typename T>
T launch_mae_compute(const T* predictions, const T* targets, size_t batch_size);

template<typename T>
void launch_mae_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size);

// Huber Loss
template<typename T>
T launch_huber_compute(const T* predictions, const T* targets, T delta, size_t batch_size);

template<typename T>
void launch_huber_gradient(const T* predictions, const T* targets, T* grad, T delta, size_t batch_size);

}

#endif
