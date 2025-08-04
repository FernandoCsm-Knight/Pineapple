#ifndef BINARY_CROSS_ENTROPY_LOSS_TPP
#define BINARY_CROSS_ENTROPY_LOSS_TPP

#include "../../inc/loss/binary_cross_entropy_loss.hpp"

#ifdef __NVCC__
#include "../../inc/device/loss_cuda_wrappers.hpp"
#endif

template <Numeric T>
T BinaryCrossEntropyLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);

#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        loss =  cuda_loss_ops::launch_binary_cross_entropy_compute<T>(predictions.data_ptr(), targets.data_ptr(), batch_size);
    } else
#endif
    {
        #pragma omp parallel for  reduction(-:loss)
        for (int i = 0; i < batch_size; ++i) {
            const T pred = std::max(std::min(predictions(i, 0).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
            const T target = targets[i]; 
            
            loss -= target * std::log(pred) + (1 - target) * std::log(1 - pred);
        }

        loss /= batch_size;
    }

    
    return loss;
}

template <Numeric T>
Tensor<T> BinaryCrossEntropyLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    Tensor<T> grad(predictions.shape());
    grad.to(predictions.device());
    const int batch_size = predictions.shape(0);

#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        cuda_loss_ops::launch_binary_cross_entropy_gradient<T>(predictions.data_ptr(), targets.data_ptr(), grad.data_ptr(), batch_size);
    } else 
#endif
    {
        #pragma omp parallel for 
        for (int i = 0; i < batch_size; ++i) {
            const T pred = std::max(std::min(predictions(i, 0).value(), static_cast<T>(1 - 1e-7)), static_cast<T>(1e-7));
            const T target = targets[i];        

            grad(i, 0) = (-target / pred + (1 - target) / (1 - pred)) / batch_size;
        }
    }
    
    return grad;
}

#endif