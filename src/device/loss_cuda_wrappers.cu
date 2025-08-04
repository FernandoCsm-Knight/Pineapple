#ifndef LOSS_CUDA_WRAPPERS_CU
#define LOSS_CUDA_WRAPPERS_CU

#include "loss_cuda_kernels.cu"
#include "../../inc/device/loss_cuda_wrappers.hpp"

namespace cuda_loss_ops {

// Helper function to perform reduction on GPU
template<typename T>
T gpu_reduce_sum(const T* input, size_t size) {
    T response = 0;

    if(size != 0) {
        T* d_temp_sum;
        dim3 grid = calculate_grid_block(size, BLOCK_SIZE);
        CUDA_CHECK(cudaMalloc(&d_temp_sum, grid.x * sizeof(T)));
        
        const size_t shared_mem_size = BLOCK_SIZE * sizeof(T);
        reduction_sum_kernel<<<grid.x, BLOCK_SIZE, shared_mem_size>>>(input, d_temp_sum, size);
        CUDA_CHECK(cudaGetLastError());

        
        if(grid.x > 1) {
            T* d_final_sum;
            CUDA_CHECK(cudaMalloc(&d_final_sum, sizeof(T)));
            reduction_sum_kernel<<<1, BLOCK_SIZE, shared_mem_size>>>(d_temp_sum, d_final_sum, grid.x);
            CUDA_CHECK(cudaGetLastError());
    
            
            T result;
            CUDA_CHECK(cudaMemcpy(&result, d_final_sum, sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_final_sum));
            CUDA_CHECK(cudaFree(d_temp_sum));
            response = result;
        } else {
            T result;
            CUDA_CHECK(cudaMemcpy(&result, d_temp_sum, sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_temp_sum));
            response = result;
        }
    } 
    
    return response;
}

// Binary Cross Entropy Loss
template<typename T>
T launch_binary_cross_entropy_compute(const T* predictions, const T* targets, size_t batch_size) {
    T* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(T)));
    
    const dim3 grid = calculate_grid_block(batch_size);
    binary_cross_entropy_compute_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, d_losses, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    const T total_loss = gpu_reduce_sum(d_losses, batch_size);
    CUDA_CHECK(cudaFree(d_losses));
    
    return total_loss / static_cast<T>(batch_size);
}

template<typename T>
void launch_binary_cross_entropy_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const dim3 grid = calculate_grid_block(batch_size);
    binary_cross_entropy_gradient_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, grad, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// Cross Entropy Loss
template<typename T>
T launch_cross_entropy_compute(const T* predictions, const T* targets, size_t batch_size, size_t num_classes) {
    T* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(T)));
    
    const dim3 grid = calculate_grid_block(batch_size);
    cross_entropy_compute_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, d_losses, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    
    const T total_loss = gpu_reduce_sum(d_losses, batch_size);
    CUDA_CHECK(cudaFree(d_losses));
    
    return total_loss / static_cast<T>(batch_size);
}

template<typename T>
void launch_cross_entropy_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size, size_t num_classes) {
    const size_t total_elements = batch_size * num_classes;
    const dim3 grid = calculate_grid_block(total_elements);
    cross_entropy_gradient_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, grad, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// MSE Loss
template<typename T>
T launch_mse_compute(const T* predictions, const T* targets, size_t batch_size) {
    T* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(T)));
    
    const dim3 grid = calculate_grid_block(batch_size);
    mse_compute_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, d_losses, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    const T total_loss = gpu_reduce_sum(d_losses, batch_size);
    CUDA_CHECK(cudaFree(d_losses));
    
    return total_loss / static_cast<T>(batch_size);
}

template<typename T>
void launch_mse_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const dim3 grid = calculate_grid_block(batch_size);
    mse_gradient_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, grad, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// MAE Loss
template<typename T>
T launch_mae_compute(const T* predictions, const T* targets, size_t batch_size) {
    T* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(T)));
    
    const dim3 grid = calculate_grid_block(batch_size);
    mae_compute_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, d_losses, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    const T total_loss = gpu_reduce_sum(d_losses, batch_size);
    CUDA_CHECK(cudaFree(d_losses));
    
    return total_loss / static_cast<T>(batch_size);
}

template<typename T>
void launch_mae_gradient(const T* predictions, const T* targets, T* grad, size_t batch_size) {
    const dim3 grid = calculate_grid_block(batch_size);
    mae_gradient_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, grad, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// Huber Loss
template<typename T>
T launch_huber_compute(const T* predictions, const T* targets, T delta, size_t batch_size) {
    T* d_losses;
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(T)));
    
    const dim3 grid = calculate_grid_block(batch_size);
    huber_compute_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, d_losses, delta, batch_size);
    CUDA_CHECK(cudaGetLastError());
    
    const T total_loss = gpu_reduce_sum(d_losses, batch_size);
    CUDA_CHECK(cudaFree(d_losses));
    
    return total_loss / static_cast<T>(batch_size);
}

template<typename T>
void launch_huber_gradient(const T* predictions, const T* targets, T* grad, T delta, size_t batch_size) {
    const dim3 grid = calculate_grid_block(batch_size);
    huber_gradient_kernel<<<grid, BLOCK_SIZE>>>(predictions, targets, grad, delta, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template float launch_binary_cross_entropy_compute<float>(const float*, const float*, size_t);
template double launch_binary_cross_entropy_compute<double>(const double*, const double*, size_t);
template void launch_binary_cross_entropy_gradient<float>(const float*, const float*, float*, size_t);
template void launch_binary_cross_entropy_gradient<double>(const double*, const double*, double*, size_t);

template float launch_cross_entropy_compute<float>(const float*, const float*, size_t, size_t);
template double launch_cross_entropy_compute<double>(const double*, const double*, size_t, size_t);
template void launch_cross_entropy_gradient<float>(const float*, const float*, float*, size_t, size_t);
template void launch_cross_entropy_gradient<double>(const double*, const double*, double*, size_t, size_t);

template float launch_mse_compute<float>(const float*, const float*, size_t);
template double launch_mse_compute<double>(const double*, const double*, size_t);
template void launch_mse_gradient<float>(const float*, const float*, float*, size_t);
template void launch_mse_gradient<double>(const double*, const double*, double*, size_t);

template float launch_mae_compute<float>(const float*, const float*, size_t);
template double launch_mae_compute<double>(const double*, const double*, size_t);
template void launch_mae_gradient<float>(const float*, const float*, float*, size_t);
template void launch_mae_gradient<double>(const double*, const double*, double*, size_t);

template float launch_huber_compute<float>(const float*, const float*, float, size_t);
template double launch_huber_compute<double>(const double*, const double*, double, size_t);
template void launch_huber_gradient<float>(const float*, const float*, float*, float, size_t);
template void launch_huber_gradient<double>(const double*, const double*, double*, double, size_t);

}

#endif
