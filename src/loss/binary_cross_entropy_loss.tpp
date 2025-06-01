#ifndef BINARY_CROSS_ENTROPY_LOSS_TPP
#define BINARY_CROSS_ENTROPY_LOSS_TPP

#include "../../inc/loss/binary_cross_entropy_loss.hpp"

template <Numeric T>
T BinaryCrossEntropyLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for  reduction(-:loss)
    for (int i = 0; i < batch_size; ++i) {
        T pred = std::max(std::min(predictions(i, 0).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
        T target = targets[i]; 
        
        loss -= target * std::log(pred) + (1 - target) * std::log(1 - pred);
    }
    
    return loss / batch_size;
}

template <Numeric T>
Tensor<T> BinaryCrossEntropyLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for 
    for (int i = 0; i < batch_size; ++i) {
        T pred = std::max(std::min(predictions(i, 0).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
        T target = targets[i];        

        grad(i, 0) = (-target / pred + (1 - target) / (1 - pred)) / batch_size;
    }
    
    return grad;
}

#endif