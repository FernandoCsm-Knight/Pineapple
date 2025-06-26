#ifndef TENSOR_FACTORY_TPP
#define TENSOR_FACTORY_TPP

#include "../../inc/tensor/tensor.hpp"

// Private Constructor

template <Numeric T>
Tensor<T>::Tensor(T* data_ptr, const Shape& shape, int* strides, bool take_ownership): Shapeable(shape) {
    data = data_ptr;
    stride = strides;
    owns_data = take_ownership;
}

// Constructors

template <Numeric T>
template <Integral... Dims>
Tensor<T>::Tensor(Dims... dims): Shapeable(dims...) {
    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;
    
    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    } 
}

template <Numeric T>
Tensor<T>::Tensor(const Shape& shape): Shapeable(shape) {
    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;

    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    } 
}

template <Numeric T>
Tensor<T>::Tensor(const Shape& shape, std::initializer_list<T> values): Shapeable(shape) {
    if(values.size() != this->length()) {
        throw std::invalid_argument("Initializer list size does not match tensor shape");
    }

    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;

    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    }

    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = *(values.begin() + i);
    }
}

template <Numeric T>
Tensor<T>::Tensor(const Shape& shape, const T& value): Shapeable(shape) {
    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;

    if(this->ndim() != 0) {
        stride[this->ndim() - 1] = 1;
        for(int i = this->ndim() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * this->shape(i + 1);
        }
    } 

    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = value;
    }
}

template <Numeric T>
Tensor<T>::Tensor(const Tensor<T>& other): Shapeable(other) { 
    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = other.data[i];
    }
}

template <Numeric T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept: Shapeable(std::move(other)) {
    data = other.data;
    stride = other.stride;
    owns_data = other.owns_data;
    other.stride = nullptr;
    other.data = nullptr;
}

template <Numeric T>
template <Numeric U>
Tensor<T>::Tensor(const Tensor<U>& other): Shapeable(other) { 
    data = new T[this->length()]();
    stride = new int[this->ndim()];
    owns_data = true;

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = static_cast<T>(other.data[i]);
    }
}

// Destructor

template <Numeric T>
Tensor<T>::~Tensor() {
    if(owns_data) {
        delete[] data;
        delete[] stride;
    }

    stride = nullptr;
    data = nullptr;
}

// Assignment operators

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
    if(this != &other) {
        if(owns_data) {
            delete[] this->data;
            delete[] stride;
    
            this->sh = other.shape();
            this->data = new T[this->length()]();

            stride = other.is_scalar() ? nullptr : new int[this->ndim()];
            for(int i = 0; i < this->ndim(); ++i) {
                stride[i] = other.stride[i];
            }
        }

        if(other.is_scalar()) {
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = other.data[0];
            }
        } else {
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = other.data[i];
            }
        }
    }

    return *this;
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& other) {
    if(this != &other) {
        if(!owns_data) {
            if(this->length() != other.length()) {
                throw std::invalid_argument("Cannot assign tensors with different sizes to a view");
            }
            
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = other.data[i];
            }
        } else {            
            delete[] this->data;
            delete[] stride;
    
            this->sh = other.shape();
            this->data = other.data;
            stride = other.stride;
            owns_data = other.owns_data;
    
            other.data = nullptr;
            other.stride = nullptr;
            other.owns_data = false;
        }
    }

    return *this;
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(const T& value) {    
    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = value;
    }
    
    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>& other) {
        if(this != &other) {
        if(owns_data) {
            delete[] this->data;
            delete[] stride;
    
            this->sh = other.shape();
            this->data = new T[this->length()]();

            stride = other.is_scalar() ? nullptr : new int[this->ndim()];
            for(int i = 0; i < this->ndim(); ++i) {
                stride[i] = other.stride[i];
            }
        }

        if(other.is_scalar()) {
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = static_cast<T>(other.data[0]);
            }
        } else {
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = static_cast<T>(other.data[i]);
            }
        }
    }

    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator=(const U& scalar) {    
    #pragma omp parallel for if(this->length() > 1000)
    for(size_t i = 0; i < this->length(); ++i) {
        data[i] = static_cast<T>(scalar);
    }
    
    return *this;
} 

// Iterators

template <Numeric T>
T* Tensor<T>::begin() {
    return this->data;
}

template <Numeric T>
T* Tensor<T>::end() {
    return this->data + this->length();
}

template <Numeric T>
const T* Tensor<T>::begin() const {
    return this->data;
}

template <Numeric T>
const T* Tensor<T>::end() const {
    return this->data + this->length();
}

#endif