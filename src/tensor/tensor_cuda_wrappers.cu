#include "../../inc/tensor/tensor_cuda_wrappers.hpp"
#include "tensor_cuda_kernels.cu"

namespace cuda_ops {

// Memory operations
template<typename T>
T* cuda_malloc(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

template<typename T>
void cuda_free(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

template<typename T>
void cuda_memcpy_host_to_device(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cuda_memcpy_device_to_host(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cuda_memcpy_device_to_device(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

// Element-wise operations
template<typename T, typename U, typename R>
void launch_tensor_add(const T* a, const U* b, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);

    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) + static_cast<R>(y); };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_subtract(const T* a, const U* b, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);

    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) - static_cast<R>(y); };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_multiply(const T* a, const U* b, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);

    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) * static_cast<R>(y); };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_divide(const T* a, const U* b, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) / static_cast<R>(y); };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Matrix multiplication
template<typename T, typename U, typename R>
void launch_tensor_matmul(const T* a, const U* b, R* result, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    tensor_matmul_kernel<<<grid, block>>>(a, b, result, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Copy
template<typename T>
void launch_tensor_copy(const T* src, T* dst, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_copy_kernel<<<grid, block>>>(src, dst, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Fill
template<typename T>
void launch_tensor_fill(T* data, T value, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_fill_kernel<<<grid, block>>>(data, value, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Scalar operations
template<typename T, typename U, typename R>
void launch_tensor_scalar_add(const T* a, U scalar, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) + static_cast<R>(y); };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_scalar_subtract(const T* a, U scalar, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) - static_cast<R>(y); };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_scalar_multiply(const T* a, U scalar, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) * static_cast<R>(y); };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U, typename R>
void launch_tensor_scalar_divide(const T* a, U scalar, R* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> R { return static_cast<R>(x) / static_cast<R>(y); };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// In-place operations
template<typename T, typename U>
void launch_tensor_inplace_add(T* a, const U* b, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x += static_cast<T>(y); };
    tensor_inplace_elementwise_kernel<<<grid, block>>>(a, b, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_subtract(T* a, const U* b, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);

    auto op = [] __device__ (T& x, const U& y) { x -= static_cast<T>(y); };
    tensor_inplace_elementwise_kernel<<<grid, block>>>(a, b, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_multiply(T* a, const U* b, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x *= static_cast<T>(y); };
    tensor_inplace_elementwise_kernel<<<grid, block>>>(a, b, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_divide(T* a, const U* b, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x /= static_cast<T>(y); };
    tensor_inplace_elementwise_kernel<<<grid, block>>>(a, b, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// In-place scalar operations
template<typename T, typename U>
void launch_tensor_inplace_scalar_add(T* a, U scalar, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x += static_cast<T>(y); };
    tensor_inplace_scalar_kernel<<<grid, block>>>(a, scalar, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_scalar_subtract(T* a, U scalar, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x -= static_cast<T>(y); };
    tensor_inplace_scalar_kernel<<<grid, block>>>(a, scalar, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_scalar_multiply(T* a, U scalar, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x *= static_cast<T>(y); };
    tensor_inplace_scalar_kernel<<<grid, block>>>(a, scalar, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_inplace_scalar_divide(T* a, U scalar, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (T& x, const U& y) { x /= static_cast<T>(y); };
    tensor_inplace_scalar_kernel<<<grid, block>>>(a, scalar, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Element-wise operations
template<typename T>
void launch_tensor_abs(const T* a, T* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_abs_kernel<<<grid, block>>>(a, result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_tensor_pow(const T* a, T* result, double exponent, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_pow_kernel<<<grid, block>>>(a, result, exponent, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_tensor_normalize(const T* a, T* result, T min_val, T max_val, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_normalize_kernel<<<grid, block>>>(a, result, min_val, max_val, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Matrix operations
template<typename T>
void launch_tensor_transpose(const T* a, T* result, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    tensor_transpose_kernel<<<grid, block>>>(a, result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_tensor_flip(const T* a, T* result, const int* shape, const int* axes,
                       const int* strides, const int* result_strides,
                       int ndim, int num_axes, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_flip_kernel<<<grid, block>>>(a, result, shape, axes, strides, result_strides, ndim, num_axes, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Reduction operations
template<typename T>
T launch_tensor_min(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(T)));
    
    tensor_min_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    if(grid_size > 1) {
        T result = launch_tensor_min(d_result, grid_size);
        CUDA_CHECK(cudaFree(d_result));
        return result;
    } else {
        T result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));
        return result;
    }
}

template<typename T>
T launch_tensor_max(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(T)));
    
    tensor_max_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    if(grid_size > 1) {
        T result = launch_tensor_max(d_result, grid_size);
        CUDA_CHECK(cudaFree(d_result));
        return result;
    } else {
        T result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));
        return result;
    }
}

template<typename T>
T launch_tensor_sum(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(T)));
    
    tensor_sum_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    if(grid_size > 1) {
        T result = launch_tensor_sum(d_result, grid_size);
        CUDA_CHECK(cudaFree(d_result));
        return result;
    } else {
        T result;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));
        return result;
    }
}

template<typename T>
T launch_tensor_norm(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(T)));
    
    tensor_norm_squared_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    T sum_of_squares;
    if(grid_size > 1) {
        sum_of_squares = launch_tensor_sum(d_result, grid_size);
    } else {
        CUDA_CHECK(cudaMemcpy(&sum_of_squares, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaFree(d_result));
    return std::sqrt(sum_of_squares);
}

// New reduction operations
template<typename T>
T launch_tensor_variance(const T* data, T mean_val, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    T* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(T)));
    
    tensor_variance_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(data, mean_val, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    T* h_result = new T[grid_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, grid_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    T final_var = T(0);
    for(int i = 0; i < grid_size; i++) {
        final_var += h_result[i];
    }
    
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_result));
    
    return final_var / T(size - 1);  // Population variance to match CPU
}

// Slice operation
template<typename T>
void launch_tensor_slice(const T* data, T* result, int start, int step, size_t new_size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(new_size, block_size);
    dim3 block(block_size);
    
    tensor_slice_kernel<<<grid, block>>>(data, result, start, step, new_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Square root operation
template<typename T>
void launch_tensor_sqrt(const T* data, T* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_sqrt_kernel<<<grid, block>>>(data, result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Boolean operations
template<typename T, typename U>
void launch_tensor_equal(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x == y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_not_equal(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x != y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_less_than(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x < y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_less_equal(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x <= y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_greater_than(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x > y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_greater_equal(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x >= y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Scalar boolean operations
template<typename T, typename U>
void launch_tensor_scalar_equal(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x == y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_not_equal(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x != y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_less_than(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x < y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_less_equal(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x <= y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_greater_than(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x > y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_greater_equal(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x >= y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Logical operations
template<typename T, typename U>
void launch_tensor_logical_and(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x && y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_logical_or(const T* a, const U* b, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x || y; };
    tensor_elementwise_kernel<<<grid, block>>>(a, b, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_logical_and(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x && y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T, typename U>
void launch_tensor_scalar_logical_or(const T* a, U scalar, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    auto op = [] __device__ (const T& x, const U& y) -> bool { return x || y; };
    tensor_scalar_kernel<<<grid, block>>>(a, scalar, result, size, op);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
void launch_tensor_logical_not(const T* a, bool* result, size_t size) {
    const int block_size = 256;
    dim3 grid = calculate_grid_block(size, block_size);
    dim3 block(block_size);
    
    tensor_logical_not_kernel<<<grid, block>>>(a, result, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
bool launch_tensor_any(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(bool)));
    
    tensor_any_kernel<<<grid_size, block_size, block_size * sizeof(bool)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    bool* h_result = new bool[grid_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, grid_size * sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool final_result = false;
    for(int i = 0; i < grid_size; i++) {
        if(h_result[i]) {
            final_result = true;
            break;
        }
    }
    
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_result));
    
    return final_result;
}

template<typename T>
bool launch_tensor_all(const T* data, size_t size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(bool)));
    
    tensor_all_kernel<<<grid_size, block_size, block_size * sizeof(bool)>>>(data, d_result, size);
    CUDA_CHECK(cudaGetLastError());
    
    bool* h_result = new bool[grid_size];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, grid_size * sizeof(bool), cudaMemcpyDeviceToHost));
    
    bool final_result = true;
    for(int i = 0; i < grid_size; i++) {
        if(!h_result[i]) {
            final_result = false;
            break;
        }
    }
    
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_result));
    
    return final_result;
}

// Explicit instantiations
template float* cuda_malloc<float>(size_t);
template void cuda_free<float>(float*);
template void cuda_memcpy_host_to_device<float>(float*, const float*, size_t);
template void cuda_memcpy_device_to_host<float>(float*, const float*, size_t);
template void cuda_memcpy_device_to_device<float>(float*, const float*, size_t);

template int* cuda_malloc<int>(size_t);
template void cuda_free<int>(int*);
template void cuda_memcpy_host_to_device<int>(int*, const int*, size_t);
template void cuda_memcpy_device_to_host<int>(int*, const int*, size_t);
template void cuda_memcpy_device_to_device<int>(int*, const int*, size_t);

template double* cuda_malloc<double>(size_t);
template void cuda_free<double>(double*);
template void cuda_memcpy_host_to_device<double>(double*, const double*, size_t);
template void cuda_memcpy_device_to_host<double>(double*, const double*, size_t);
template void cuda_memcpy_device_to_device<double>(double*, const double*, size_t);

template bool* cuda_malloc<bool>(size_t);
template void cuda_free<bool>(bool*);
template void cuda_memcpy_host_to_device<bool>(bool*, const bool*, size_t);
template void cuda_memcpy_device_to_host<bool>(bool*, const bool*, size_t);
template void cuda_memcpy_device_to_device<bool>(bool*, const bool*, size_t);

// Arithmetic operations
template void launch_tensor_add<float, float, float>(const float*, const float*, float*, size_t);
template void launch_tensor_subtract<float, float, float>(const float*, const float*, float*, size_t);
template void launch_tensor_multiply<float, float, float>(const float*, const float*, float*, size_t);
template void launch_tensor_divide<float, float, float>(const float*, const float*, float*, size_t);
template void launch_tensor_matmul<float, float, float>(const float*, const float*, float*, int, int, int);
template void launch_tensor_copy<float>(const float*, float*, size_t);
template void launch_tensor_fill<float>(float*, float, size_t);

// Cross-type operations
template void launch_tensor_add<int, float, float>(const int*, const float*, float*, size_t);

// Operations for int
template void launch_tensor_copy<int>(const int*, int*, size_t);
template void launch_tensor_fill<int>(int*, int, size_t);

// Operations for double  
template void launch_tensor_copy<double>(const double*, double*, size_t);
template void launch_tensor_fill<double>(double*, double, size_t);

// Scalar operations
template void launch_tensor_scalar_add<float, float, float>(const float*, float, float*, size_t);
template void launch_tensor_scalar_subtract<float, float, float>(const float*, float, float*, size_t);
template void launch_tensor_scalar_multiply<float, float, float>(const float*, float, float*, size_t);
template void launch_tensor_scalar_divide<float, float, float>(const float*, float, float*, size_t);

// In-place operations
template void launch_tensor_inplace_add<float, float>(float*, const float*, size_t);
template void launch_tensor_inplace_subtract<float, float>(float*, const float*, size_t);
template void launch_tensor_inplace_multiply<float, float>(float*, const float*, size_t);
template void launch_tensor_inplace_divide<float, float>(float*, const float*, size_t);

template void launch_tensor_inplace_scalar_add<float, float>(float*, float, size_t);
template void launch_tensor_inplace_scalar_subtract<float, float>(float*, float, size_t);
template void launch_tensor_inplace_scalar_multiply<float, float>(float*, float, size_t);
template void launch_tensor_inplace_scalar_divide<float, float>(float*, float, size_t);

// Reduction operations
template float launch_tensor_min<float>(const float*, size_t);
template float launch_tensor_max<float>(const float*, size_t);
template float launch_tensor_sum<float>(const float*, size_t);
template float launch_tensor_norm<float>(const float*, size_t);

// Boolean operations
template void launch_tensor_equal<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_not_equal<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_less_than<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_less_equal<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_greater_than<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_greater_equal<float, float>(const float*, const float*, bool*, size_t);

template void launch_tensor_scalar_equal<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_not_equal<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_less_than<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_less_equal<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_greater_than<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_greater_equal<float, float>(const float*, float, bool*, size_t);

template void launch_tensor_logical_and<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_logical_or<float, float>(const float*, const float*, bool*, size_t);
template void launch_tensor_scalar_logical_and<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_scalar_logical_or<float, float>(const float*, float, bool*, size_t);
template void launch_tensor_logical_not<float>(const float*, bool*, size_t);
template bool launch_tensor_any<float>(const float*, size_t);
template bool launch_tensor_all<float>(const float*, size_t);

template void launch_tensor_logical_and<bool, bool>(const bool*, const bool*, bool*, size_t);
template void launch_tensor_logical_or<bool, bool>(const bool*, const bool*, bool*, size_t);
template void launch_tensor_scalar_logical_and<bool, bool>(const bool*, bool, bool*, size_t);
template void launch_tensor_scalar_logical_or<bool, bool>(const bool*, bool, bool*, size_t);
template void launch_tensor_logical_not<bool>(const bool*, bool*, size_t);
template bool launch_tensor_any<bool>(const bool*, size_t);
template bool launch_tensor_all<bool>(const bool*, size_t);

// New operations
template float launch_tensor_variance<float>(const float*, float, size_t);
template void launch_tensor_slice<float>(const float*, float*, int, int, size_t);
template void launch_tensor_sqrt<float>(const float*, float*, size_t);

// Element-wise operations
template void launch_tensor_abs<float>(const float*, float*, size_t);
template void launch_tensor_abs<int>(const int*, int*, size_t);
template void launch_tensor_abs<double>(const double*, double*, size_t);

template void launch_tensor_pow<float>(const float*, float*, double, size_t);
template void launch_tensor_pow<int>(const int*, int*, double, size_t);
template void launch_tensor_pow<double>(const double*, double*, double, size_t);

template void launch_tensor_normalize<float>(const float*, float*, float, float, size_t);
template void launch_tensor_normalize<int>(const int*, int*, int, int, size_t);
template void launch_tensor_normalize<double>(const double*, double*, double, double, size_t);

// Matrix operations
template void launch_tensor_transpose<float>(const float*, float*, int, int);
template void launch_tensor_transpose<int>(const int*, int*, int, int);
template void launch_tensor_transpose<double>(const double*, double*, int, int);

template void launch_tensor_flip<float>(const float*, float*, const int*, const int*, const int*, const int*, int, int, size_t);
template void launch_tensor_flip<int>(const int*, int*, const int*, const int*, const int*, const int*, int, int, size_t);
template void launch_tensor_flip<double>(const double*, double*, const int*, const int*, const int*, const int*, int, int, size_t);

}
