#ifndef MSE_LOSS_TPP
#define MSE_LOSS_TPP

#include "../../inc/loss/mse_loss.hpp"

#ifdef __NVCC__
#include "../../inc/device/loss_cuda_wrappers.hpp"
#endif

template <Numeric T>
T MSELoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);

#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        loss = cuda_loss_ops::launch_mse_compute<T>(predictions.data_ptr(), targets.data_ptr(), batch_size);
    } else 
#endif
    {
        #pragma omp parallel for reduction(+:loss)
        for (int i = 0; i < batch_size; ++i) {
            const T diff = predictions(i, 0).value() - targets[i];
            loss += diff * diff;
        }

        loss /= batch_size;
    }

    return loss;
}

template <Numeric T>
Tensor<T> MSELoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    grad.to(predictions.device());
    const int batch_size = predictions.shape(0);
    
#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        cuda_loss_ops::launch_mse_gradient<T>(predictions.data_ptr(), targets.data_ptr(), grad.data_ptr(), batch_size);
    } else
#endif
    {   
        #pragma omp parallel for
        for(int i = 0; i < batch_size; ++i) {
            grad(i, 0) = 2 * (predictions(i, 0).value() - targets[i]) / batch_size;
        }
    }    
    
    return grad;
}

#endif