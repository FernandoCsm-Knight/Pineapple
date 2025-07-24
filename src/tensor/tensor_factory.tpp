#ifndef TENSOR_FACTORY_TPP
#define TENSOR_FACTORY_TPP

#include "../../inc/tensor/tensor.hpp"

#ifdef PINEAPPLE_CUDA_ENABLED
#include "../../inc/tensor/tensor_cuda_wrappers.hpp"
#endif

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
    device = other.device;

#ifdef PINEAPPLE_CUDA_ENABLED
    if (device == Device::GPU) {
        data = cuda_ops::cuda_malloc<T>(this->length());
        cuda_ops::cuda_memcpy_device_to_device(data, other.data, this->length());
    } else
#endif
    {
        data = new T[this->length()]();
        #pragma omp parallel for if(this->length() > 1000)
        for(size_t i = 0; i < this->length(); ++i) {
            data[i] = other.data[i];
        }
    }

    stride = new int[this->ndim()];
    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }
}

template <Numeric T>
Tensor<T>::Tensor(Tensor<T>&& other) noexcept: Shapeable(std::move(other)) {
    data = other.data;
    stride = other.stride;
    owns_data = other.owns_data;
    device = other.device;
    other.stride = nullptr;
    other.data = nullptr;
}

template <Numeric T>
template <Numeric U>
Tensor<T>::Tensor(const Tensor<U>& other): Shapeable(other) { 
    device = other.device;

#ifdef PINEAPPLE_CUDA_ENABLED
    if (device == Device::GPU) {
        data = cuda_ops::cuda_malloc<T>(this->length());
    } else
#endif
    {
        data = new T[this->length()]();
    }
    
    stride = new int[this->ndim()];

    for(int i = 0; i < this->ndim(); ++i) {
        stride[i] = other.stride[i];
    }

#ifdef PINEAPPLE_CUDA_ENABLED
    if (device == Device::GPU) {
        cuda_ops::launch_tensor_type_convert(other.data, data, this->length());
    } else
#endif
    {
        #pragma omp parallel for if(this->length() > 1000)
        for(size_t i = 0; i < this->length(); ++i) {
            data[i] = static_cast<T>(other.data[i]);
        }
    }
}

// Destructor

template <Numeric T>
Tensor<T>::~Tensor() {
    if(owns_data) {
#ifdef PINEAPPLE_CUDA_ENABLED
        if (device == Device::GPU) {
            cuda_ops::cuda_free(data);
        } else
#endif
        {
            delete[] data;
        }
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
#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU) {
                cuda_ops::cuda_free(this->data);
            } else
#endif
            {
                delete[] this->data;
            }
            delete[] stride;
    
            this->sh = other.shape();
            device = other.device;

#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU) {
                this->data = cuda_ops::cuda_malloc<T>(this->length());
            } else
#endif
            {
                this->data = new T[this->length()]();
            }

            stride = other.is_scalar() ? nullptr : new int[this->ndim()];
            for(int i = 0; i < this->ndim(); ++i) {
                stride[i] = other.stride[i];
            }
        }

        if(other.is_scalar()) {
#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU) {
                if (other.device == Device::GPU) {
                    T scalar_value;
                    cuda_ops::cuda_memcpy_device_to_host(&scalar_value, other.data, 1);
                    cuda_ops::launch_tensor_fill(this->data, scalar_value, this->length());
                } else {
                    cuda_ops::launch_tensor_fill(this->data, other.data[0], this->length());
                }
            } else
#endif
            {
                T scalar_value;
#ifdef PINEAPPLE_CUDA_ENABLED
                if (other.device == Device::GPU) {
                    cuda_ops::cuda_memcpy_device_to_host(&scalar_value, other.data, 1);
                } else
#endif
                {
                    scalar_value = other.data[0];
                }
                
                #pragma omp parallel for if(this->length() > 1000)
                for(size_t i = 0; i < this->length(); ++i) {
                    this->data[i] = scalar_value;
                }
            }
        } else {
#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU && other.device == Device::GPU) {
                cuda_ops::cuda_memcpy_device_to_device(this->data, other.data, this->length());
            } else if (device == Device::GPU && other.device == Device::CPU) {
                cuda_ops::cuda_memcpy_host_to_device(this->data, other.data, this->length());
            } else if (device == Device::CPU && other.device == Device::GPU) {
                cuda_ops::cuda_memcpy_device_to_host(this->data, other.data, this->length());
            } else
#endif
            {
                #pragma omp parallel for if(this->length() > 1000)
                for(size_t i = 0; i < this->length(); ++i) {
                    this->data[i] = other.data[i];
                }
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
            
#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU && other.device == Device::GPU) {
                cuda_ops::cuda_memcpy_device_to_device(this->data, other.data, this->length());
            } else if (device == Device::GPU && other.device == Device::CPU) {
                cuda_ops::cuda_memcpy_host_to_device(this->data, other.data, this->length());
            } else if (device == Device::CPU && other.device == Device::GPU) {
                cuda_ops::cuda_memcpy_device_to_host(this->data, other.data, this->length());
            } else
#endif
            {
                #pragma omp parallel for if(this->length() > 1000)
                for(size_t i = 0; i < this->length(); ++i) {
                    this->data[i] = other.data[i];
                }
            }
        } else {            
#ifdef PINEAPPLE_CUDA_ENABLED
            if (device == Device::GPU) {
                cuda_ops::cuda_free(this->data);
            } else
#endif
            {
                delete[] this->data;
            }
            delete[] stride;
    
            // Move resources
            this->sh = other.shape();
            this->data = other.data;
            stride = other.stride;
            owns_data = other.owns_data;
            device = other.device;
    
            other.data = nullptr;
            other.stride = nullptr;
            other.owns_data = false;
        }
    }

    return *this;
}

template <Numeric T>
Tensor<T>& Tensor<T>::operator=(const T& value) {
#ifdef PINEAPPLE_CUDA_ENABLED
    if (device == Device::GPU) {
        cuda_ops::launch_tensor_fill(data, value, this->length());
    } else
#endif
    {
        #pragma omp parallel for if(this->length() > 1000)
        for(size_t i = 0; i < this->length(); ++i) {
            data[i] = value;
        }
    }
    
    return *this;
}

template <Numeric T>
template <Numeric U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>& other) {
    if(owns_data) {
#ifdef PINEAPPLE_CUDA_ENABLED
        if (device == Device::GPU) {
            cuda_ops::cuda_free(this->data);
        } else
#endif
        {
            delete[] this->data;
        }

        delete[] stride;
        this->sh = other.shape();
        
#ifdef PINEAPPLE_CUDA_ENABLED
        if (device == Device::GPU) {
            this->data = cuda_ops::cuda_malloc<T>(this->length());
        } else
#endif
        {
            this->data = new T[this->length()]();
        }

        stride = other.is_scalar() ? nullptr : new int[this->ndim()];
        for(int i = 0; i < this->ndim(); ++i) {
            stride[i] = other.stride[i];
        }
    }

    if(other.is_scalar()) {
        T scalar_value;
#ifdef PINEAPPLE_CUDA_ENABLED
        if (other.device == Device::GPU) {
            U gpu_scalar;
            cuda_ops::cuda_memcpy_device_to_host(&gpu_scalar, other.data, 1);
            scalar_value = static_cast<T>(gpu_scalar);
        } else
#endif
        {
            scalar_value = static_cast<T>(other.data[0]);
        }
        
#ifdef PINEAPPLE_CUDA_ENABLED
        if (device == Device::GPU) {
            cuda_ops::launch_tensor_fill(this->data, scalar_value, this->length());
        } else
#endif
        {
            #pragma omp parallel for if(this->length() > 1000)
            for(size_t i = 0; i < this->length(); ++i) {
                this->data[i] = scalar_value;
            }
        }
    } else {
#ifdef PINEAPPLE_CUDA_ENABLED
        if (device == Device::GPU && other.device == Device::GPU) {
            if constexpr (std::is_same_v<T, U>) {
                cuda_ops::cuda_memcpy_device_to_device(this->data, other.data, this->length());
            } else {
                cuda_ops::launch_tensor_type_convert(other.data, this->data, this->length());
            }
        } else if (device == Device::GPU && other.device == Device::CPU) {
            if constexpr (std::is_same_v<T, U>) {
                cuda_ops::cuda_memcpy_host_to_device(this->data, other.data, this->length());
            } else {
                T* temp_data = new T[this->length()];
                #pragma omp parallel for if(this->length() > 1000)
                for(size_t i = 0; i < this->length(); ++i) {
                    temp_data[i] = static_cast<T>(other.data[i]);
                }
                cuda_ops::cuda_memcpy_host_to_device(this->data, temp_data, this->length());
                delete[] temp_data;
            }
        } else if (device == Device::CPU && other.device == Device::GPU) {
            if constexpr (std::is_same_v<T, U>) {
                cuda_ops::cuda_memcpy_device_to_host(this->data, other.data, this->length());
            } else {
                U* temp_data = new U[this->length()];
                cuda_ops::cuda_memcpy_device_to_host(temp_data, other.data, this->length());
                #pragma omp parallel for if(this->length() > 1000)
                for(size_t i = 0; i < this->length(); ++i) {
                    this->data[i] = static_cast<T>(temp_data[i]);
                }
                delete[] temp_data;
            }
        } else
#endif
        {
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
    const T converted_scalar = static_cast<T>(scalar);
    
#ifdef PINEAPPLE_CUDA_ENABLED
    if (device == Device::GPU) {
        cuda_ops::launch_tensor_fill(data, converted_scalar, this->length());
    } else
#endif
    {
        #pragma omp parallel for if(this->length() > 1000)
        for(size_t i = 0; i < this->length(); ++i) {
            data[i] = converted_scalar;
        }
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