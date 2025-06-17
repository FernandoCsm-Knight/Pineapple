#ifndef MAE_LOSS_TPP
#define MAE_LOSS_TPP

#include "../../inc/loss/mae_loss.hpp"

template <Numeric T>
T MAELoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);
    
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < batch_size; ++i) {
        loss += std::abs(predictions(i, 0).value() - targets[i]);
    }
    
    return loss / batch_size;
}

template <Numeric T>
Tensor<T> MAELoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    const int batch_size = predictions.shape(0);

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        const T diff = predictions(i, 0).value() - targets[i];
        grad(i, 0) = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) / batch_size;
    }
    
    return grad;
}

#endif