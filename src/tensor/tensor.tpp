#ifndef TENSOR_TPP
#define TENSOR_TPP

#include "../../inc/tensor/tensor.hpp"

// Helper methods

template <Numeric T>
void Tensor<T>::append(const Tensor<T>& other, int axis) {
    if (axis < 0 || axis >= this->ndim()) {
        throw std::invalid_argument("Invalid axis for append operation");
    }
    
    for (int i = 0; i < this->ndim(); ++i) {
        if (i != axis && this->shape(i) != other.shape(i)) {
            throw std::invalid_argument("Tensors must have the same shape except along the append axis");
        }
    }
    
    Shape new_shape = this->shape();
    new_shape.resize_dimension(axis, this->shape(axis) + other.shape(axis));
    
    Tensor<T> result(new_shape);
    
    std::vector<int> idx(this->ndim(), 0);
    for (size_t i = 0; i < this->length(); ++i) {
        size_t remaining = i;
        for (int d = 0; d < this->ndim(); ++d) {
            idx[d] = remaining / this->stride[d];
            remaining %= this->stride[d];
        }
        
        size_t result_idx = 0;
        for (int d = 0; d < this->ndim(); ++d) {
            result_idx += idx[d] * result.stride[d];
        }
        
        result.data[result_idx] = this->data[i];
    }
    
    std::vector<int> other_idx(other.ndim(), 0);
    for (size_t i = 0; i < other.length(); ++i) {
        size_t remaining = i;
        for (int d = 0; d < other.ndim(); ++d) {
            other_idx[d] = remaining / other.stride[d];
            remaining %= other.stride[d];
        }
        
        std::vector<int> result_idx = other_idx;
        result_idx[axis] += this->shape(axis);
        
        size_t linear_idx = 0;
        for (int d = 0; d < result.ndim(); ++d) {
            linear_idx += result_idx[d] * result.stride[d];
        }
        
        result.data[linear_idx] = other.data[i];
    }
    
    *this = std::move(result);
}

template <Numeric T>
void Tensor<T>::remove(int axis) {
    if (axis < 0 || axis >= this->ndim()) {
        throw std::invalid_argument("Invalid axis for remove operation");
    }
    
    if (this->shape(axis) == 1) {
        Shape new_shape;
        for (int i = 0; i < this->ndim(); ++i) {
            if (i != axis) {
                new_shape.add_dimension(this->shape(i));
            }
        }
        
        Tensor<T> result(new_shape);
        
        std::vector<int> idx(this->ndim(), 0);
        std::vector<int> new_idx(new_shape.ndim(), 0);
        
        for (size_t i = 0; i < this->length(); ++i) {
            size_t remaining = i;
            for (int d = 0; d < this->ndim(); ++d) {
                idx[d] = remaining / this->stride[d];
                remaining %= this->stride[d];
            }
            
            int j = 0;
            for (int d = 0; d < this->ndim(); ++d) {
                if (d != axis) {
                    new_idx[j++] = idx[d];
                }
            }
            
            size_t result_idx = 0;
            for (int d = 0; d < new_shape.ndim(); ++d) {
                result_idx += new_idx[d] * result.stride[d];
            }
            
            result.data[result_idx] = this->data[i];
        }
        
        *this = std::move(result);
    } else {
        throw std::invalid_argument("Removal of non-singleton dimensions not implemented. Use slice instead.");
    }
}

template <Numeric T>
T* Tensor<T>::data_ptr() const {
    return this->data;
}

template <Numeric T>
T Tensor<T>::min() const {
    T min_element = this->data[0];

    #pragma omp parallel for reduction(min:min_element)
    for(size_t i = 1; i < this->length(); ++i) {
        min_element = std::min(min_element, this->data[i]);
    }

    return min_element;
}

template <Numeric T>
T Tensor<T>::max() const {
    T max_element = this->data[0];

    #pragma omp parallel for reduction(max:max_element)
    for(size_t i = 1; i < this->length(); ++i) {
        max_element = std::max(max_element, this->data[i]);
    }

    return max_element;
}

template <Numeric T>
Tensor<T> Tensor<T>::normilize() const {
    T min_val = this->min();
    T max_val = this->max();
    
    Tensor<T> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = (this->data[i] - min_val) / (max_val - min_val);
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::argmax(int axis) const {
    if(axis < 0 || axis >= this->ndim()) {
        throw std::invalid_argument("Invalid axis value for argmax operation");
    }
    
    Shape result_shape;
    for(int i = 0; i < this->ndim(); ++i) {
        if(i != axis) {
            result_shape.add_dimension(this->shape(i));
        }
    }
    
    Tensor<T> result(result_shape);
    
    int axis_stride = 1;
    for(int i = this->ndim() - 1; i > axis; --i) {
        axis_stride *= this->shape(i);
    }

    int axis_size = this->shape(axis);
    int outer_stride = 1;
    for(int i = 0; i < axis; ++i) {
        outer_stride *= this->shape(i);
    }
    
    #pragma omp parallel for collapse(2)
    for(int outer = 0; outer < outer_stride; ++outer) {
        for(int inner = 0; inner < axis_stride; ++inner) {
            T max_idx = 0;
            T max_val = this->data[outer * axis_size * axis_stride + inner];
            
            for(int a = 1; a < axis_size; ++a) {
                size_t idx = outer * axis_size * axis_stride + a * axis_stride + inner;
                if(this->data[idx] > max_val) {
                    max_val = this->data[idx];
                    max_idx = a;
                }
            }
            
            result.data[outer * axis_stride + inner] = max_idx;
        }
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::sum(int axis, bool keep_dimension) const {
    Tensor<T> result;

    if(axis == -1) {
        T total = 0;

        if(keep_dimension) {
            Shape result_shape;
            for(int i = 0; i < this->ndim(); ++i) {
                result_shape.add_dimension(1);
            }
            
            result = Tensor<T>(result_shape);
        } 

        #pragma omp parallel for reduction(+:total)
        for(size_t i = 0; i < this->length(); ++i) {
            total += this->data[i];
        }

        result[0] = total;
    } else {        
        if(axis < 0 || axis >= this->ndim()) {
            throw std::invalid_argument("Invalid axis value for sum operation");
        }
        
        Shape result_shape;
        for(int i = 0; i < this->ndim(); ++i) {
            if(i == axis) {
                if (keep_dimension) result_shape.add_dimension(1);
            } else {
                result_shape.add_dimension(this->shape()[i]);
            }
        }
        
        result = Tensor<T>(result_shape);
        
        int axis_stride = 1;
        for(int i = this->ndim() - 1; i > axis; --i) {
            axis_stride *= this->shape()[i];
        }
    
        int axis_size = this->shape()[axis];
        int outer_block_size = 1;
        for(int i = 0; i < axis; ++i) {
            outer_block_size *= this->shape()[i];
        }
        
        #pragma omp parallel for collapse(2)
        for(int outer = 0; outer < outer_block_size; ++outer) {
            for(int inner = 0; inner < axis_stride; ++inner) {
                size_t result_idx = (keep_dimension) ?
                    outer * axis_stride * (result_shape[axis]) + inner :
                    outer * axis_stride + inner;
                
                for(int a = 0; a < axis_size; ++a) {
                    size_t src_idx = outer * axis_size * axis_stride + a * axis_stride + inner;
                    result[result_idx] += this->data[src_idx];
                }
            }
        }
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::pow(double exponent) const {
    Tensor<T> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = std::pow(this->data[i], exponent);
    }

    return result;
}

template <Numeric T>
T Tensor<T>::mean() const {
    return sum() / this->length();
}

template <Numeric T>
T Tensor<T>::var() const {
    T sum = 0;
    T mean_value = mean();

    #pragma omp parallel for reduction(+:sum)
    for(size_t i = 0; i < this->length(); ++i) {
        sum += (this->data[i] - mean_value) * (this->data[i] - mean_value);
    }

    return sum / this->length();
}

template <Numeric T>
T Tensor<T>::std() const {
    return std::sqrt(var());
}

template <Numeric T>
Tensor<T> Tensor<T>::abs() const {
    Tensor<T> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = std::abs(this->data[i]);
    }

    return result;
}

template <Numeric T>
void Tensor<T>::fill(const T& value) {
    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        this->data[i] = value;
    }
}

// Formatted output

template <Numeric T>
Tensor<T> Tensor<T>::reshape(Shape new_shape) const {
    if(new_shape.length() != this->length()) {
        throw std::invalid_argument("New shape must have the same number of elements as the original shape");
    }

    Tensor<T> result(new_shape);

    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = this->data[i];
    }

    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::flatten() const {
    return reshape(Shape(this->length()));
}

template <Numeric T>
Tensor<T> Tensor<T>::slice(int start, int end, int step) const {
    if(start < 0 || end > this->shape(0)) {
        throw std::invalid_argument("Slice indices out of bounds");
    }

    int new_size = (end - start) / step;
    Tensor<T> result(new_size);

    #pragma omp parallel for
    for(int i = 0; i < new_size; ++i) {
        result.data[i] = this->data[start + i * step];
    }

    return result;
}

template <Numeric T>
std::set<T> Tensor<T>::unique() const {
    std::set<T> unique;

    for(size_t i = 0; i < this->length(); ++i) {
        unique.insert(this->data[i]);
    }

    return unique;
}

template <Numeric T>
Tensor<T> Tensor<T>::clip(const T& min_val, const T& max_val) const {
    Tensor<T> result(this->shape());

    #pragma omp parallel for
    for(size_t i = 0; i < this->length(); ++i) {
        result.data[i] = std::clamp(this->data[i], min_val, max_val);
    }

    return result;
}

#endif