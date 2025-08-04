#ifndef TENSOR_CUDA_TPP
#define TENSOR_CUDA_TPP

#include "../../inc/tensor/tensor.hpp"

#ifdef __NVCC__
#include "../../inc/device/tensor_cuda_wrappers.hpp"
#endif

// Device management methods

template <Numeric T>
void Tensor<T>::to(Device target_device) {
    if(device() != target_device) {
        #ifdef __NVCC__
            if(target_device == Device::GPU) {
                T* gpu_data = cuda_ops::cuda_malloc<T>(this->length());
                cuda_ops::cuda_memcpy_host_to_device(gpu_data, data, this->length());
                
                if(owns_data) delete[] data;
                
                data = gpu_data;
                this->current_device = Device::GPU;
                owns_data = true;
            } else {
                T* cpu_data = new T[this->length()];
                cuda_ops::cuda_memcpy_device_to_host(cpu_data, data, this->length());
                
                if(owns_data) cuda_ops::cuda_free(data);
                
                data = cpu_data;
                this->current_device = Device::CPU;
                owns_data = true;
            }
        #else
            if(target_device == Device::GPU) {
                throw std::runtime_error("CUDA support not compiled. Please compile with nvcc.");
            }
        #endif
    }
}

#endif
