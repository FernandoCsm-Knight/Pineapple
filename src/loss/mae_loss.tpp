#ifndef MAE_LOSS_TPP
#define MAE_LOSS_TPP

#include "../../inc/loss/mae_loss.hpp"

#ifdef __NVCC__
#include "../../inc/device/loss_cuda_wrappers.hpp"
#endif

template <Numeric T>
T MAELoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);

#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        loss = cuda_loss_ops::launch_mae_compute<T>(predictions.data_ptr(), targets.data_ptr(), batch_size);
    } else
#endif
    {
        #pragma omp parallel for reduction(+:loss)
        for (int i = 0; i < batch_size; ++i) {
            loss += std::abs(predictions(i, 0).value() - targets[i]);
        }

        loss /= batch_size;
    }

    return loss;
}

template <Numeric T>
Tensor<T> MAELoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    grad.to(predictions.device());
    const int batch_size = predictions.shape(0);
    
#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        cuda_loss_ops::launch_mae_gradient<T>(predictions.data_ptr(), targets.data_ptr(), grad.data_ptr(), batch_size);
    } else
#endif
    {
        #pragma omp parallel for
        for (int i = 0; i < batch_size; ++i) {
            const T diff = predictions(i, 0).value() - targets[i];
            grad(i, 0) = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) / batch_size;
        }
    }
    
    return grad;
}

#endif