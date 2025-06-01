#ifndef BATCH_NORMALIZATION_TPP
#define BATCH_NORMALIZATION_TPP

#include "../../inc/layer/batch_normalization.hpp"

template <Numeric T>
BatchNormalization<T>::BatchNormalization(int num_features, T epsilon, T momentum) {
    this->epsilon = epsilon;
    this->momentum = momentum;
    this->is_training = true;
    
    Shape param_shape{num_features};
    this->gamma = Tensor<T>(param_shape, 1.0);
    this->beta = Tensor<T>(param_shape, 0.0);
    this->running_mean = Tensor<T>(param_shape, 0.0);
    this->running_var = Tensor<T>(param_shape, 1.0);
}

template <Numeric T>
void BatchNormalization<T>::train() {
    this->is_training = true;
}

template <Numeric T>
void BatchNormalization<T>::eval() {
    this->is_training = false;
}

template <Numeric T>
Tensor<T> BatchNormalization<T>::forward(const Tensor<T>& input) {
    this->input_shape = input.shape();
    this->input = input;
    
    int batch_size = 1;
    int channels = input.shape(0);
    int spatial_size = 1;
    
    for (int i = 1; i < input.ndim(); i++) {
        spatial_size *= input.shape(i);
    }
    
    if (is_training) {
        mean = Tensor<T>({channels}, 0.0);
        
        #pragma omp parallel for
        for (int c = 0; c < channels; c++) {
            T sum = 0;
            for (int i = 0; i < spatial_size; i++) {
                sum += input(c, i / input.shape(2), i % input.shape(2)).value();
            }
            mean(c) = sum / spatial_size;
        }
        
        var = Tensor<T>({channels}, 0.0);
        
        #pragma omp parallel for
        for (int c = 0; c < channels; c++) {
            T sum_sq = 0;
            for (int i = 0; i < spatial_size; i++) {
                T diff = input(c, i / input.shape(2), i % input.shape(2)).value() - mean(c).value();
                sum_sq += diff * diff;
            }
            var(c) = sum_sq / spatial_size;
        }
        
        #pragma omp parallel for
        for (int c = 0; c < channels; c++) {
            running_mean(c) = running_mean(c).value() * momentum + mean(c).value() * (1 - momentum);
            running_var(c) = running_var(c).value() * momentum + var(c).value() * (1 - momentum);
        }
    }
    
    std_dev = Tensor<T>({channels});
    
    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        T variance = is_training ? var(c).value() : running_var(c).value();
        std_dev(c) = std::sqrt(variance + epsilon);
    }
    
    normalized = Tensor<T>(input_shape);
    Tensor<T> output(input_shape);
    
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < input.shape(1); h++) {
            for (int w = 0; w < input.shape(2); w++) {
                T mean_val = is_training ? mean(c).value() : running_mean(c).value();
                normalized(c, h, w) = (input(c, h, w).value() - mean_val) / std_dev(c).value();
                output(c, h, w) = gamma(c).value() * normalized(c, h, w).value() + beta(c).value();
            }
        }
    }
    
    return output;
}

template <Numeric T>
Tensor<T> BatchNormalization<T>::backward(const Tensor<T>& grad_output) {
    if (!is_training) {
        throw std::runtime_error("BatchNormalization backward should only be called in training mode");
    }
    
    int batch_size = 1;
    int channels = input.shape(0);
    int spatial_size = 1;
    
    for (int i = 1; i < input.ndim(); i++) {
        spatial_size *= input.shape(i);
    }
    
    Tensor<T> dgamma({channels}, 0.0);
    Tensor<T> dbeta({channels}, 0.0);
    
    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        T dg_sum = 0;
        T db_sum = 0;
        
        for (int i = 0; i < spatial_size; i++) {
            int h = i / input.shape(2);
            int w = i % input.shape(2);
            
            dg_sum += grad_output(c, h, w).value() * normalized(c, h, w).value();
            db_sum += grad_output(c, h, w).value();
        }
        
        dgamma(c) = dg_sum;
        dbeta(c) = db_sum;
    }
    
    Tensor<T> grad_input(input_shape);
    
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < input.shape(1); h++) {
            for (int w = 0; w < input.shape(2); w++) {
                T dx_hat = grad_output(c, h, w).value() * gamma(c).value();
                
                T dvar = -0.5 * dx_hat * (input(c, h, w).value() - mean(c).value()) * 
                          std::pow(var(c).value() + epsilon, -1.5);
                
                T dmean = -dx_hat / std::sqrt(var(c).value() + epsilon);
                dmean -= 2 * dvar * (input(c, h, w).value() - mean(c).value()) / spatial_size;
                
                grad_input(c, h, w) = dx_hat / std::sqrt(var(c).value() + epsilon) + 
                                     dvar * 2 * (input(c, h, w).value() - mean(c).value()) / spatial_size + 
                                     dmean / spatial_size;
            }
        }
    }
    
    update_gamma(dgamma);
    update_beta(dbeta);
    
    return grad_input;
}

#endif 