#ifndef DEVICE_CONFIG_HPP
#define DEVICE_CONFIG_HPP

#include <string>
#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pineapple {
    enum Device {
        CPU, CUDA
    };
    
    extern Device current_device;
    
    void set_device(Device device);
    Device get_device();

    void set_device(const std::string& device_name);
    Device get_device(const std::string& device_name);

    bool is_device_available(Device device);
    bool is_device_available(const std::string& device_name);

    bool is_cuda_available();
    bool is_cpu_available();
}

#include "../../src/device/device_config.tpp"

#endif