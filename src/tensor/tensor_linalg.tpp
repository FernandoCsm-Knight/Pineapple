#ifndef TENSOR_LINALG
#define TENSOR_LINALG

#include "../../inc/tensor/tensor.hpp"

template <Numeric T>
T Tensor<T>::norm() const {
    T sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(size_t i = 0; i < this->length(); ++i) {
        sum += this->data[i] * this->data[i];
    }

    return std::sqrt(sum);
}

template <Numeric T>
Tensor<T> Tensor<T>::dilate(int size, std::vector<int> axes) const {
    if(axes.empty()) {
        for(int i = 0; i < this->ndim(); ++i) {
            axes.push_back(i);
        }
    } else {
        for(int axis : axes) {
            if(axis < 0 || axis >= this->ndim()) {
                throw std::invalid_argument("Invalid axis for dilation operation");
            }
        }
    }
    
    Shape new_shape = this->shape();
    for(int axis : axes) {
        new_shape.resize_dimension(axis, this->shape(axis) + (this->shape(axis) - 1) * size);
    }
    
    Tensor<T> result(new_shape, T(0));
    
    std::vector<int> input_idx(this->ndim(), 0);
    std::vector<int> output_idx(this->ndim(), 0);
    
    for(size_t flat_idx = 0; flat_idx < this->length(); ++flat_idx) {
        size_t remaining = flat_idx;
        for(int i = 0; i < this->ndim(); ++i) {
            input_idx[i] = remaining / this->stride[i];
            remaining %= this->stride[i];
        }
        
        for(int i = 0; i < this->ndim(); ++i) {
            if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
                output_idx[i] = input_idx[i] * (size + 1);
            } else {
                output_idx[i] = input_idx[i];
            }
        }
        
        size_t result_idx = 0;
        for(int i = 0; i < this->ndim(); ++i) {
            result_idx += output_idx[i] * result.stride[i];
        }
        
        result.data[result_idx] = this->data[flat_idx];
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::pad(int size, std::vector<int> axes) const {
    if(axes.empty()) {
        for(int i = 0; i < this->ndim(); ++i) {
            axes.push_back(i);
        }
    } else {
        for(int axis : axes) {
            if(axis < 0 || axis >= this->ndim()) {
                throw std::invalid_argument("Invalid axis for padding operation");
            }
        }
    }
    
    Shape new_shape = this->shape();
    for(int axis : axes) {
        new_shape.resize_dimension(axis, this->shape(axis) + 2 * size);
    }
    
    Tensor<T> result(new_shape, T(0));
    
    std::vector<int> input_idx(this->ndim(), 0);
    std::vector<int> output_idx(this->ndim(), 0);
    
    for(size_t flat_idx = 0; flat_idx < this->length(); ++flat_idx) {
        size_t remaining = flat_idx;
        for(int i = 0; i < this->ndim(); ++i) {
            input_idx[i] = remaining / this->stride[i];
            remaining %= this->stride[i];
        }
        
        for(int i = 0; i < this->ndim(); ++i) {
            if(std::find(axes.begin(), axes.end(), i) != axes.end()) {
                output_idx[i] = input_idx[i] + size;
            } else {
                output_idx[i] = input_idx[i];
            }
        }
        
        size_t result_idx = 0;
        for(int i = 0; i < this->ndim(); ++i) {
            result_idx += output_idx[i] * result.stride[i];
        }
        
        result.data[result_idx] = this->data[flat_idx];
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::flip(std::vector<int> axes) const {
    if(axes.empty()) {
        for(int i = 0; i < this->ndim(); ++i) {
            axes.push_back(i);
        }
    } else {
        for(int axis : axes) {
            if(axis < 0 || axis >= this->ndim()) {
                throw std::invalid_argument("Invalid axis for flip operation");
            }
        }
    }
    
    Tensor<T> result(this->shape());
    
    std::vector<int> idx_input(this->ndim(), 0);
    std::vector<int> idx_output(this->ndim(), 0);
    
    for(size_t flat_idx = 0; flat_idx < this->length(); ++flat_idx) {
        size_t remaining = flat_idx;
        for(int i = 0; i < this->ndim(); ++i) {
            idx_input[i] = remaining / this->stride[i];
            remaining %= this->stride[i];
        }
        
        idx_output = idx_input;
        
        for(int axis : axes) {
            idx_output[axis] = this->shape(axis) - 1 - idx_input[axis];
        }
        
        size_t linear_input = 0;
        size_t linear_output = 0;
        for(int i = 0; i < this->ndim(); ++i) {
            linear_input += idx_input[i] * this->stride[i];
            linear_output += idx_output[i] * result.stride[i];
        }
        
        result.data[linear_output] = this->data[linear_input];
    }
    
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::cross_correlation(const Tensor<U>& kernel, int stride, Correlation mode, int padding) const {
    if(this->ndim() < 2 || kernel.ndim() < 2) {
        throw std::invalid_argument("Both input tensor and kernel must have at least 2 dimensions");
    } else if(kernel.ndim() != 2 && kernel.ndim() != 3) {
        throw std::invalid_argument("Kernel must be 2D or 3D");
    }
    
    int batch_size = 1, channels = 1,
        input_h = this->shape(this->ndim() - 2),
        input_w = this->shape(this->ndim() - 1);
    
    if(this->ndim() == 3) {
        channels = this->shape(0);
    } else if(this->ndim() == 4) {
        batch_size = this->shape(0);
        channels = this->shape(1);
    }
    
    int kernel_h = kernel.shape(kernel.ndim() - 2),
        kernel_w = kernel.shape(kernel.ndim() - 1);

    if(kernel.ndim() == 3 && kernel.shape(0) != channels && kernel.shape(0) != 1) {        
        throw std::invalid_argument("Kernel channels must match input channels or be 1");
    }  
    
    switch(mode) {
        case Correlation::same:
            padding = (kernel.shape(kernel.ndim() - 2) - 1) / 2;
            break;

        case Correlation::full:
            padding = kernel.shape(kernel.ndim() - 2) - 1;
            break;

        case Correlation::valid:
        default:
            break;
    }

    int output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    
    Shape output_shape(output_h, output_w);
    if(this->ndim() == 3) {
        output_shape.insert_dimension(0, channels);
    } else if(this->ndim() == 4) {
        output_shape.insert_dimension(0, channels);
        output_shape.insert_dimension(0, batch_size);
    }

    using R = std::common_type_t<T, U>;
    
    Tensor<R> result(output_shape);
    #pragma omp parallel for collapse(4)
    for(int b = 0; b < batch_size; ++b) {
        for(int c = 0; c < channels; ++c) {
            for(int i = 0; i < output_h; ++i) {
                for(int j = 0; j < output_w; ++j) {
                    R sum = 0;
                    
                    // Aplica o kernel
                    for(int ki = 0; ki < kernel_h; ++ki) {
                        for(int kj = 0; kj < kernel_w; ++kj) {
                            // Posição no tensor de entrada com offset de padding
                            int di = i * stride + ki - padding;
                            int dj = j * stride + kj - padding;
                            
                            // Verifica se a posição está dentro dos limites
                            if(di >= 0 && di < input_h && dj >= 0 && dj < input_w) {
                                // Calcular os índices corretamente com base na dimensionalidade
                                size_t input_idx;
                                size_t kernel_idx;
                                
                                // Acesso ao tensor de entrada
                                if (this->ndim() == 2) {
                                    input_idx = di * this->stride[0] + dj;
                                } else if (this->ndim() == 3) {
                                    input_idx = c * this->stride[0] + di * this->stride[1] + dj;
                                } else { // ndim == 4
                                    input_idx = b * this->stride[0] + c * this->stride[1] + di * this->stride[2] + dj;
                                }
                                
                                // Acesso ao kernel
                                if(kernel.ndim() == 2) {
                                    kernel_idx = ki * kernel.stride[0] + kj;
                                } else { // ndim == 3
                                    // Usar canal específico do kernel ou o canal 0 se o kernel tiver apenas 1 canal
                                    int kernel_c = (kernel.shape(0) == 1) ? 0 : c;
                                    kernel_idx = kernel_c * kernel.stride[0] + ki * kernel.stride[1] + kj;
                                }
                                
                                sum += static_cast<R>(this->data[input_idx]) * static_cast<R>(kernel.data[kernel_idx]);
                            }
                        }
                    }
                    
                    // Calcular o índice de saída
                    size_t output_idx;
                    if(this->ndim() == 2) {
                        output_idx = i * result.stride[0] + j;
                    } else if(this->ndim() == 3) {
                        output_idx = c * result.stride[0] + i * result.stride[1] + j;
                    } else {
                        output_idx = b * result.stride[0] + c * result.stride[1] + i * result.stride[2] + j;
                    }
                    
                    result.data[output_idx] = sum;
                }
            }
        }
    }
    
    return result;
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::convolve(const Tensor<U>& kernel, int stride, Correlation mode, int padding) const {
    return cross_correlation(kernel.flip(), stride, mode, padding);
}

template <Numeric T>
template <Numeric U>
Tensor<std::common_type_t<T, U>> Tensor<T>::dot(const Tensor<U>& other) const {
    if (this->ndim() > 2 || other.ndim() > 2) {
        throw std::invalid_argument("Tensors must have at most 2 dimensions");
    }
    
    int m = (this->ndim() == 1) ? 1 : this->shape(0);
    int n = (this->ndim() == 1) ? this->shape(0) : this->shape(1);
    int p = (other.ndim() == 1) ? 1 : other.shape(1);
    
    if ((this->ndim() == 2 && other.ndim() == 2 && this->shape(1) != other.shape(0)) ||
        (this->ndim() == 1 && other.ndim() == 2 && this->shape(0) != other.shape(0)) ||
        (this->ndim() == 2 && other.ndim() == 1 && this->shape(1) != other.shape(0))) {
        throw std::invalid_argument("Cannot multiply tensors with incompatible shapes");
    }
    
    using R = std::common_type_t<T, U>;
    
    Shape result_shape;
    if (this->ndim() == 1 && other.ndim() == 1) {
        // Produto escalar: retorna shape()
        result_shape = Shape();
    } else if (this->ndim() == 1 && other.ndim() == 2) {
        // Vetor x Matrix: (n,) x (n, p) = (p,)
        result_shape = Shape(p);
    } else if (this->ndim() == 2 && other.ndim() == 1) {
        // Matrix x Vetor: (m, n) x (n,) = (m,)
        result_shape = Shape(m);
    } else {
        // Matrix x Matrix: (m, n) x (n, p) = (m, p)
        // NUNCA remover dimensões singleton aqui
        result_shape = Shape(m, p);
    }
    
    Tensor<R> result(result_shape);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            R sum = 0;
            
            for (int k = 0; k < n; ++k) {
                int idx1 = (this->ndim() == 1) ? k : i * stride[0] + k * stride[1];
                int idx2 = (other.ndim() == 1) ? k : k * other.stride[0] + j * other.stride[1];
                
                sum += static_cast<R>(this->data[idx1]) * static_cast<R>(other.data[idx2]);
            }
            
            if (this->ndim() == 1 && other.ndim() == 2) {
                // Vetor x Matrix: (n,) x (n, p) = (p,)
                result.data[j] = sum;
            } else if (this->ndim() == 2 && other.ndim() == 1) {
                // Matrix x Vetor: (m, n) x (n,) = (m,)
                result.data[i] = sum;
            } else {
                // Matrix x Matrix: (m, n) x (n, p) = (m, p)
                result.data[i * result.stride[0] + j * result.stride[1]] = sum;
            }
        }
    }
    
    return result;
}

template <Numeric T>
Tensor<T> Tensor<T>::transpose() const {
    if(this->ndim() != 2) {
        throw std::invalid_argument("Tensor must have 2 dimensions");
    }

    Tensor<T> result(this->shape(1), this->shape(0));

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < this->shape(0); ++i) {
        for(int j = 0; j < this->shape(1); ++j) {
            result.data[j * result.stride[0] + i] = this->data[i * this->stride[0] + j];
        }
    }

    return result;
}

#endif