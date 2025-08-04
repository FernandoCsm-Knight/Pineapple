#ifndef ACTIVATION_CUDA_WRAPPERS_CU
#define ACTIVATION_CUDA_WRAPPERS_CU

#include "activation_cuda_kernels.cu"
#include "../../inc/device/activation_cuda_wrappers.hpp"

namespace cuda_activation_ops {

template<typename T>
void launch_relu_apply(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    relu_apply_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_relu_derivative(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    relu_derivative_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_sigmoid_apply(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    sigmoid_apply_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_sigmoid_derivative(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    sigmoid_derivative_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_tanh_apply(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    tanh_apply_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_tanh_derivative(const T* input, T* output, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    tanh_derivative_kernel<<<grid, BLOCK_SIZE>>>(input, output, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_elu_apply(const T* input, T* output, T alpha, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    elu_apply_kernel<<<grid, BLOCK_SIZE>>>(input, output, alpha, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_elu_derivative(const T* input, T* output, T alpha, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    elu_derivative_kernel<<<grid, BLOCK_SIZE>>>(input, output, alpha, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_leaky_relu_apply(const T* input, T* output, T alpha, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    leaky_relu_apply_kernel<<<grid, BLOCK_SIZE>>>(input, output, alpha, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_leaky_relu_derivative(const T* input, T* output, T alpha, size_t size) {
    const dim3 grid = calculate_grid_block(size);
    
    leaky_relu_derivative_kernel<<<grid, BLOCK_SIZE>>>(input, output, alpha, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_softmax_apply(const T* input, T* output, int batch_size, int num_classes, size_t size) {
    const int block = min(1024, num_classes);
    const int grid = batch_size;
    
    softmax_apply_kernel<<<grid, block>>>(input, output, batch_size, num_classes, size);
    
    CUDA_CHECK(cudaGetLastError());
}

template void launch_relu_apply<float>(const float*, float*, size_t);
template void launch_relu_apply<double>(const double*, double*, size_t);
template void launch_relu_derivative<float>(const float*, float*, size_t);
template void launch_relu_derivative<double>(const double*, double*, size_t);

template void launch_sigmoid_apply<float>(const float*, float*, size_t);
template void launch_sigmoid_apply<double>(const double*, double*, size_t);
template void launch_sigmoid_derivative<float>(const float*, float*, size_t);
template void launch_sigmoid_derivative<double>(const double*, double*, size_t);

template void launch_tanh_apply<float>(const float*, float*, size_t);
template void launch_tanh_apply<double>(const double*, double*, size_t);
template void launch_tanh_derivative<float>(const float*, float*, size_t);
template void launch_tanh_derivative<double>(const double*, double*, size_t);

template void launch_elu_apply<float>(const float*, float*, float, size_t);
template void launch_elu_apply<double>(const double*, double*, double, size_t);
template void launch_elu_derivative<float>(const float*, float*, float, size_t);
template void launch_elu_derivative<double>(const double*, double*, double, size_t);

template void launch_leaky_relu_apply<float>(const float*, float*, float, size_t);
template void launch_leaky_relu_apply<double>(const double*, double*, double, size_t);
template void launch_leaky_relu_derivative<float>(const float*, float*, float, size_t);
template void launch_leaky_relu_derivative<double>(const double*, double*, double, size_t);

template void launch_softmax_apply<float>(const float*, float*, int, int, size_t);
template void launch_softmax_apply<double>(const double*, double*, int, int, size_t);

}

#endif
