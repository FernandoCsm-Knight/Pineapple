#ifndef MSE_LOSS_TPP
#define MSE_LOSS_TPP

#include "../../inc/loss/mse_loss.hpp"

template <Numeric T>
T MSELoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);
    
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < batch_size; ++i) {
        const T diff = predictions(i, 0).value() - targets[i];
        loss += diff * diff;
    }
    
    return loss / batch_size;
}

template <Numeric T>
Tensor<T> MSELoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    const int batch_size = predictions.shape(0);
    
    #pragma omp parallel for
    for(int i = 0; i < batch_size; ++i) {
        grad(i, 0) = 2 * (predictions(i, 0).value() - targets[i]) / batch_size;
    }
    
    return grad;
}

#endif