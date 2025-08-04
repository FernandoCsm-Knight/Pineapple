#ifndef TENSOR_ACCESS_TPP
#define TENSOR_ACCESS_TPP

#include "../../inc/tensor/tensor.hpp"

template <Numeric T>
template <Integral... Indices>
Tensor<T> Tensor<T>::operator()(Indices... indices) {
    const int dims = sizeof...(indices);
    const size_t offset = this->index(indices...);
    
    Shape subview_shape;
    int* subview_stride = nullptr;
    if(dims != this->ndim()) {
        subview_stride = this->stride + dims;
        
        for(int i = dims; i < this->ndim(); ++i) 
            subview_shape.add_dimension(this->shape(i));
    }

    Tensor<T> result(this->data + offset, subview_shape, subview_stride);
    result.current_device = this->device();
    return result;
}

template <Numeric T>
template <Integral... Indices>
Tensor<T> Tensor<T>::operator()(Indices... indices) const {
    const int dims = sizeof...(indices);
    const size_t offset = this->index(indices...);
    
    Shape subview_shape;
    int* subview_stride = nullptr;
    if(dims != this->ndim()) {
        subview_stride = this->stride + dims;
        
        for(int i = dims; i < this->ndim(); ++i) 
            subview_shape.add_dimension(this->shape(i));
    }

    Tensor<T> result(this->data + offset, subview_shape, subview_stride);
    result.current_device = this->device();
    return result;
}

template <Numeric T>
template <Integral... Indices>
size_t Tensor<T>::index(Indices... indices) const {
    const int dims = sizeof...(indices);
    if (dims > this->ndim()) {
        throw std::invalid_argument("Too many indices for tensor dimensions");
    }
    
    const int idx_array[] = { static_cast<int>(indices)... };
    
    size_t offset = 0;
    for(int i = 0; i < dims; ++i) {
        if(idx_array[i] < 0 || idx_array[i] >= this->shape(i)) {
            std::stringstream ss;
            ss << "Array index " << i << " out of bounds for dimension " << this->shape(i);
            throw std::out_of_range(ss.str());
        }
        
        offset += idx_array[i] * stride[i];
    }
    
    return offset;
}

template <Numeric T>
T& Tensor<T>::operator[](size_t idx) {
    return data[idx];
}

template <Numeric T>
const T& Tensor<T>::operator[](size_t idx) const {
    return data[idx];
}

template <Numeric T>
T& Tensor<T>::value() {
    if(this->ndim() != 0)
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

template <Numeric T>
const T& Tensor<T>::value() const {
    if(this->ndim() != 0)
        throw std::invalid_argument("value() can only be called on scalar (0D) views");
    
    return *data;
}

#endif 
