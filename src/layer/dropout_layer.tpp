#ifndef DROPOUT_LAYER_TPP
#define DROPOUT_LAYER_TPP

#include "../../inc/layer/dropout_layer.hpp"

template <Numeric T>
Dropout<T>::Dropout(T rate): dropout_fraction(rate) {
    if (rate < 0.0 || rate >= 1.0) {
        throw std::invalid_argument("Dropout rate must be in range [0.0, 1.0)");
    }
}

template <Numeric T>
Tensor<T> Dropout<T>::forward(const Tensor<T>& input) {
    Tensor<T> result = input;

    if(this->is_in_train_mode() && dropout_fraction > 0.0) {
        mask = Tensor<bool>(input.shape());
        
        std::random_device rd;
        
        #pragma omp parallel for
        for(size_t i = 0; i < mask.length(); ++i) {
            thread_local std::mt19937 gen(rd() + omp_get_thread_num());
            std::bernoulli_distribution dist(1.0 - dropout_fraction);
            mask[i] = dist(gen);
        }
    
        Tensor<T> output(input.shape());
        const T scale = static_cast<T>(1.0) / (static_cast<T>(1.0) - dropout_fraction);
        
        #pragma omp parallel for
        for(size_t i = 0; i < input.length(); ++i) {
            output[i] = mask[i] ? input[i] * scale : static_cast<T>(0);
        }
        
        result = output;
    }

    return result;
}

template <Numeric T>
Tensor<T> Dropout<T>::backward(const Tensor<T>& grad_output) {
    Tensor<T> result = grad_output;

    if(this->is_in_train_mode() && dropout_fraction > 0.0) {
        Tensor<T> grad_input(grad_output.shape());
        const T scale = static_cast<T>(1.0) / (static_cast<T>(1.0) - dropout_fraction);
        
        #pragma omp parallel for
        for(size_t i = 0; i < grad_output.length(); ++i) {
            grad_input[i] = mask[i] ? grad_output[i] * scale : static_cast<T>(0);
        }
        
        result = grad_input;
    }

    return result;
}

template <Numeric T>
bool Dropout<T>::is_activation() const {
    return true; 
}

template <Numeric T>
bool Dropout<T>::is_optimization() const {
    return false; 
}

template <Numeric T>
T Dropout<T>::dropout_rate() const {
    return dropout_fraction;
}

#endif
