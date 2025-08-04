#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <functional>
#include <type_traits>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cmath>
#include <omp.h>

#include "../types/numeric.hpp"
#include "../types/device.hpp"
#include "../abstract/shapeable.hpp"
#include "../../inc/abstract/support_device.hpp"

template <Numeric T> class Tensor: public Shapeable, public SupportDevice {
    template <Numeric> friend class Tensor;

    private:
        T* data = nullptr;
        int* stride = nullptr;
        bool owns_data = true;

        // Private methods

        size_t get_broadcast_index(size_t i, const std::vector<size_t>& other_shape, const std::vector<size_t>& other_stride) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> simd_with_tensor(
            const Tensor<U>& other,
            std::function<void(std::common_type_t<T, U>&, const T&, const U&)> callback
        ) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> simd_with_scalar(
            const U& scalar,
            std::function<void(std::common_type_t<T, U>&, const T&, const U&)> callback
        ) const;

        template <Numeric U>
        Tensor<T>& change_tensor_simd(
            const Tensor<U>& other, 
            std::function<void(T&, const U&)> callback
        );

        template <Numeric U>
        Tensor<T>& change_tensor_scalar_simd(
            const U& scalar, 
            std::function<void(T&, const U&)> callback
        );

#ifdef __NVCC__
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> cuda_binary_op(
            const Tensor<U>& other,
            void (*cuda_kernel)(const T*, const U*, std::common_type_t<T, U>*, size_t)
        ) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> cuda_scalar_op(
            const U& scalar,
            void (*cuda_kernel)(const T*, U, std::common_type_t<T, U>*, size_t)
        ) const;

        template <Numeric U>
        Tensor<T>& cuda_inplace_tensor_op(
            const Tensor<U>& other,
            void (*cuda_kernel)(T*, const U*, size_t)
        );

        template <Numeric U>
        Tensor<T>& cuda_inplace_scalar_op(
            const U& scalar,
            void (*cuda_kernel)(T*, U, size_t)
        );

        template <Numeric U>
        Tensor<bool> cuda_comparison_op(
            const Tensor<U>& other,
            void (*cuda_kernel)(const T*, const U*, bool*, size_t)
        ) const;

        template <Numeric U>
        Tensor<bool> cuda_scalar_comparison_op(
            const U& scalar,
            void (*cuda_kernel)(const T*, U, bool*, size_t)
        ) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> cuda_broadcast_op(
            const Tensor<U>& other,
            int operation
        ) const;

        bool cuda_reduction_op(
            bool (*cuda_kernel)(const T*, size_t)
        ) const;
#endif

        // Private Constructor

        Tensor(T* data_ptr, const Shape& shape, int* strides, bool take_ownership = false);

    public:

        // Constructors
    
        template<Integral... Dims>
        Tensor(Dims... dims);

        Tensor(const Shape& shape);

        Tensor(const Shape& shape, std::initializer_list<T> values);

        Tensor(const Shape& shape, const T& value);

        Tensor(const Tensor<T>& other);

        Tensor(Tensor<T>&& other) noexcept;
        
        template <Numeric U>
        Tensor(const Tensor<U>& other);

        // Destructor

        ~Tensor();

        // Assingment operators

        Tensor<T>& operator=(const Tensor<T>& other);
        Tensor<T>& operator=(Tensor<T>&& other);
        Tensor<T>& operator=(const T& value);

        template <Numeric U>
        Tensor<T>& operator=(const Tensor<U>& other);

        template <Numeric U>
        Tensor<T>& operator+=(const Tensor<U>& other);
        
        template <Numeric U>
        Tensor<T>& operator-=(const Tensor<U>& other);

        template <Numeric U>
        Tensor<T>& operator*=(const Tensor<U>& other);

        template <Numeric U>
        Tensor<T>& operator/=(const Tensor<U>& other);

        template <Numeric U>
        Tensor<T>& operator=(const U& scalar);
        
        template <Numeric U>
        Tensor<T>& operator+=(const U& scalar);
        
        template <Numeric U>
        Tensor<T>& operator-=(const U& scalar);

        template <Numeric U>
        Tensor<T>& operator*=(const U& scalar);

        template <Numeric U>
        Tensor<T>& operator/=(const U& scalar);

        // Arithmetic operators

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator+(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator-(const Tensor<U>& other) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator*(const Tensor<U>& other) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator/(const Tensor<U>& other) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator+(const U& scalar) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator-(const U& scalar) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator*(const U& scalar) const;
        
        template <Numeric U>
        Tensor<std::common_type_t<T, U>> operator/(const U& scalar) const;

        // Boolean operators

        template <Numeric U>
        Tensor<bool> operator==(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator!=(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator<(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator<=(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator>(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator>=(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator==(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator!=(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator<(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator<=(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator>(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator>=(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator||(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator&&(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<bool> operator||(const U& scalar) const;

        template <Numeric U>
        Tensor<bool> operator&&(const U& scalar) const;

        Tensor<bool> operator!() const;

        // Boolean methods

        bool any() const;
        bool all() const;
        
        // Accessors

        template<Integral... Indices>
        Tensor<T> operator()(Indices... indices);

        template<Integral... Indices>
        Tensor<T> operator()(Indices... indices) const; 

        T& operator[](size_t idx);
        const T& operator[](size_t idx) const;

        T& value();
        const T& value() const;

        template <Integral... Indices>
        size_t index(Indices... indices) const;

        // Helper methods

        T* data_ptr() const;

        T min() const;
        T max() const;
        T norm() const;
        T mean() const;
        T var() const;
        T std() const;
        
        void fill(const T& value);

        std::set<T> unique() const;
        Tensor<T> clip(const T& min_value, const T& max_value) const;
        Tensor<T> argmax(int axis = 0) const;
        Tensor<T> sum(int axis = -1, bool keep_dimension = false) const;
        Tensor<T> pow(double exponent) const;
        Tensor<T> normalize() const;
        Tensor<T> abs() const;

        void append(const Tensor<T>& other, int axis = 0);
        void remove(int axis = 0);

        Tensor<T> flip(std::vector<int> axes = {}) const;
        Tensor<T> dilate(int size, std::vector<int> axes = {}) const;
        Tensor<T> pad(int size, std::vector<int> axes = {}) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> dot(const Tensor<U>& other) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> cross_correlation(const Tensor<U>& kernel, int stride = 1, Correlation mode = Correlation::valid, int padding = 0) const;

        template <Numeric U>
        Tensor<std::common_type_t<T, U>> convolve(const Tensor<U>& kernel, int stride = 1, Correlation mode = Correlation::valid, int padding = 0) const;

        // Device management
        
        void to(Device target_device) override;

        // Formatted tensors

        Tensor<T> transpose() const;
        Tensor<T> reshape(Shape new_shape) const;
        Tensor<T> squeeze() const;
        Tensor<T> unsqueeze(int idx) const;
        Tensor<T> flatten() const;
        Tensor<T> slice(int start, int end, int step = 1) const;

        // Iterators

        T* begin();
        T* end();
        const T* begin() const;
        const T* end() const;

        // Formatted output

        friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
            if(tensor.is_scalar()) {
                os << "Tensor(" << tensor.value() << ")";
            } else {
                os << "Tensor(" << std::endl;
                os << "[";
                for(size_t i = 0; i < tensor.length(); ++i) {
                    for(int j = 0;  j < tensor.ndim() - 1; ++j) {
                        if(i % tensor.stride[j] == 0) {
                            os << "[";
                        }
                    }
    
                    os << tensor.data[i];
                
                    bool end_line = false;
                    for(int j = 0;  j < tensor.ndim() - 1; ++j) {
                        if((i + 1) % tensor.stride[j] == 0) {
                            os << "]";
                            end_line = true;
                        }
                    }
                    
                    if (i < tensor.length() - 1) {
                        os << ", ";
                        if(end_line) {
                            os << std::endl;
                        }
                    }
                }
                os << "]" << std::endl;
                os << tensor.shape() << std::endl << ")";
            }

            return os;
        }
};

#include "../../src/tensor/tensor.tpp"
#include "../../src/tensor/tensor_factory.tpp"
#include "../../src/tensor/tensor_arithmetic.tpp"
#include "../../src/tensor/tensor_simd.tpp"
#include "../../src/tensor/tensor_access.tpp"
#include "../../src/tensor/tensor_boolean.tpp"
#include "../../src/tensor/tensor_linalg.tpp"
#include "../../src/tensor/tensor_cuda.tpp"

#endif 