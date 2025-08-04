#ifndef CROSS_ENTROPY_LOSS_TPP
#define CROSS_ENTROPY_LOSS_TPP

#include "../../inc/loss/cross_entropy_loss.hpp"

#ifdef __NVCC__
#include "../../inc/device/loss_cuda_wrappers.hpp"
#endif

template <Numeric T>
T CrossEntropyLoss<T>::compute(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    T loss = 0;
    const int batch_size = predictions.shape(0);

#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        const int num_classes = predictions.shape(1);
        loss = cuda_loss_ops::launch_cross_entropy_compute<T>(predictions.data_ptr(), targets.data_ptr(), batch_size, num_classes);
    } else
#endif
    {
        #pragma omp parallel for reduction(-:loss)
        for(int i = 0; i < batch_size; ++i) {
            const int target_idx = static_cast<int>(targets[i]);
            const T pred_prob = std::max(predictions(i, target_idx).value(), static_cast<T>(1e-7));
            loss -= std::log(pred_prob);
        }

        loss /= batch_size;
    }    

    return loss;
}

template <Numeric T>
Tensor<T> CrossEntropyLoss<T>::gradient(const Tensor<T>& predictions, const Tensor<T>& targets) const {
    const int batch_size = predictions.shape(0);
    const int num_classes = predictions.shape(1);
    Tensor<T> grad(predictions.shape());
    grad.to(predictions.device());
    
#ifdef __NVCC__
    if(predictions.is_cuda() && targets.is_cuda()) {
        cuda_loss_ops::launch_cross_entropy_gradient<T>(predictions.data_ptr(), targets.data_ptr(), grad.data_ptr(), batch_size, num_classes);
    } else
#endif
    {
        #pragma omp parallel for
        for(int i = 0; i < batch_size; ++i) {
            const int target_idx = static_cast<int>(targets[i]);
            
            for(int j = 0; j < num_classes; ++j) {
                grad(i, j) = predictions(i, j).value();
                if(j == target_idx) grad(i, j) -= 1;
            }
        }

        grad /= batch_size;
    }
    

    return grad;
}

#endif