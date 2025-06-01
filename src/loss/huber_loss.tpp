#ifndef HUBER_LOSS_TPP
#define HUBER_LOSS_TPP

#include "../../inc/loss/huber_loss.hpp"

template <Numeric T>
T HuberLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for reduction(+:loss)
    for(int i = 0; i < batch_size; ++i) {
        T diff = std::abs(predictions(i, 0).value() - targets[i]);
        
        if(diff < delta) {
            loss += 0.5 * diff * diff;
        } else {
            loss += delta * (diff - 0.5 * delta);
        }
    }
    
    return loss / batch_size;
}

template <Numeric T>
Tensor<T> HuberLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    int batch_size = predictions.shape(0);
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        T diff = predictions(i, 0).value() - targets[i];
        T abs_diff = std::abs(diff);
        
        if (abs_diff < delta) {
            grad(i, 0) = diff / batch_size;
        } else {
            grad(i, 0) = delta * (diff > 0 ? 1 : -1) / batch_size;
        }
    }
    
    return grad;
}

#endif