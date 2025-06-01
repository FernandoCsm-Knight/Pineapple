#ifndef DEVICE_CONFIG_TPP
#define DEVICE_CONFIG_TPP

#include "../../inc/device/device_config.hpp"

using namespace pineapple;

Device pineapple::current_device = Device::CPU;

Device pineapple::get_device() {
    return current_device;
}

void pineapple::set_device(Device device) {
    current_device = device;
}

Device pineapple::get_device(const std::string& device_name) {
    std::string lower = device_name;
    for(char& c : lower) {
        c = std::tolower(c);
    }

    Device device;
    if (lower == "cpu") {
        device = Device::CPU;
    } else if (lower == "cuda") {
        device = Device::CUDA;
    } else {
        throw std::invalid_argument("Invalid device name: " + device_name);
    }

    return device;
}

void pineapple::set_device(const std::string& device_name) {
    set_device(get_device(device_name));
}

bool pineapple::is_device_available(Device device) {
    bool available = false;

    if (device == Device::CPU) {
        available = is_cpu_available();
    } else if (device == Device::CUDA) {
        available = is_cuda_available();
    }

    return available;
}

bool pineapple::is_device_available(const std::string& device_name) {
    return is_device_available(get_device(device_name));
}

bool pineapple::is_cuda_available() {
    bool available = false;

    #ifdef CUDA_ENABLED 
        try {
            int device_count = 0;
        
            cudaError_t err = cudaGetDeviceCount(&device_count);
            available = (err == cudaSuccess && device_count > 0);

            if (available) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "CUDA device: " << prop.name << std::endl;
            } else {
                std::cout << "No CUDA devices available." << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "CUDA error: " << e.what() << std::endl;
        }
    #endif

    return available;
}

bool pineapple::is_cpu_available() {
    return true; 
}

#endif