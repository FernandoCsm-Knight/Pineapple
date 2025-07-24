#ifndef TENSOR_CUDA_WRAPPERS_HPP
#define TENSOR_CUDA_WRAPPERS_HPP

#include <cuda_runtime.h>
#include <type_traits>
#include <stdexcept>
#include <iostream>

namespace cuda_ops {

// Macro para verificar erros CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

// Função auxiliar para calcular grid e block dimensions
inline dim3 calculate_grid_block(size_t size, int block_size = 256) {
    const int grid_size = (size + block_size - 1) / block_size;
    return dim3(grid_size, 1, 1);
}

// Declarações das funções wrapper (implementações em .cu)
template<typename T, typename U, typename R>
void launch_tensor_add(const T* a, const U* b, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_subtract(const T* a, const U* b, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_multiply(const T* a, const U* b, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_divide(const T* a, const U* b, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_scalar_add(const T* a, U scalar, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_scalar_subtract(const T* a, U scalar, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_scalar_multiply(const T* a, U scalar, R* result, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_scalar_divide(const T* a, U scalar, R* result, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_add(T* a, const U* b, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_subtract(T* a, const U* b, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_multiply(T* a, const U* b, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_divide(T* a, const U* b, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_scalar_add(T* a, U scalar, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_scalar_subtract(T* a, U scalar, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_scalar_multiply(T* a, U scalar, size_t size);

template<typename T, typename U>
void launch_tensor_inplace_scalar_divide(T* a, U scalar, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_broadcast(
    const T* a, const U* b, R* result,
    const size_t* a_strides, const size_t* b_strides,
    const size_t* result_strides, const int* shape,
    size_t total_elements, int ndim, int operation
);

template<typename T>
void launch_tensor_fill(T* data, T value, size_t size);

template<typename T>
void launch_tensor_copy(const T* src, T* dst, size_t size);

template<typename T, typename U, typename R>
void launch_tensor_matmul(const T* a, const U* b, R* result, int M, int N, int K);

// Reduction operations
template<typename T>
T launch_tensor_min(const T* data, size_t size);

template<typename T>
T launch_tensor_max(const T* data, size_t size);

template<typename T>
T launch_tensor_sum(const T* data, size_t size);

template<typename T>
T launch_tensor_norm(const T* data, size_t size);

// Element-wise operations
template<typename T>
void launch_tensor_pow(const T* a, T* result, double exponent, size_t size);

template<typename T>
void launch_tensor_abs(const T* a, T* result, size_t size);

template<typename T>
void launch_tensor_clip(const T* a, T* result, T min_val, T max_val, size_t size);

template<typename T>
void launch_tensor_normalize(const T* a, T* result, T min_val, T max_val, size_t size);

// Matrix operations
template<typename T>
void launch_tensor_transpose(const T* a, T* result, int rows, int cols);

template<typename T>
void launch_tensor_flip(const T* a, T* result, const int* shape, const int* axes,
                       const int* strides, const int* result_strides,
                       int ndim, int num_axes, size_t size);

// Boolean comparison operations
template<typename T, typename U>
void launch_tensor_equal(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_not_equal(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_less_than(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_less_equal(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_greater_than(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_greater_equal(const T* a, const U* b, bool* result, size_t size);

// Boolean scalar comparison operations
template<typename T, typename U>
void launch_tensor_scalar_equal(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_not_equal(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_less_than(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_less_equal(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_greater_than(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_greater_equal(const T* a, U scalar, bool* result, size_t size);

// Logical operations
template<typename T, typename U>
void launch_tensor_logical_and(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_logical_or(const T* a, const U* b, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_logical_and(const T* a, U scalar, bool* result, size_t size);

template<typename T, typename U>
void launch_tensor_scalar_logical_or(const T* a, U scalar, bool* result, size_t size);

template<typename T>
void launch_tensor_logical_not(const T* a, bool* result, size_t size);

// Reduction operations for boolean
template<typename T>
bool launch_tensor_any(const T* data, size_t size);

template<typename T>
bool launch_tensor_all(const T* data, size_t size);

// Sum reduction operation
template<typename T>
T launch_tensor_sum(const T* data, size_t size);

// Variance operation
template<typename T>
T launch_tensor_variance(const T* data, T mean_val, size_t size);

// Argmax operation
template<typename T>
int launch_tensor_argmax(const T* data, size_t size);

// Slice operation
template<typename T>
void launch_tensor_slice(const T* data, T* result, int start, int step, size_t new_size);

// Square root operation
template<typename T>
void launch_tensor_sqrt(const T* data, T* result, size_t size);

// Funções auxiliares para alocação e cópia de memória
template<typename T>
T* cuda_malloc(size_t size);

template<typename T>
void cuda_free(T* ptr);

template<typename T>
void cuda_memcpy_host_to_device(T* dst, const T* src, size_t size);

template<typename T>
void cuda_memcpy_device_to_host(T* dst, const T* src, size_t size);

template<typename T>
void cuda_memcpy_device_to_device(T* dst, const T* src, size_t size);

}

#endif
