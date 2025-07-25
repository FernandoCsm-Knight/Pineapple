#include "../../inc/device/cuda_detector.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

#if __NVCC__
    #include <cuda_runtime.h>
#endif

namespace pineapple {

    bool is_cuda_available() {
#if __NVCC__
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess && device_count > 0);
#else
        return false;
#endif
    }

    int cuda_device_count() {
#if __NVCC__
        int device_count = 0;
        if(is_cuda_available()) cudaGetDeviceCount(&device_count);
        return device_count;
#else
        return 0;
#endif
    }

    CudaDeviceInfo cuda_device_info(int device_id) {
#if __NVCC__
        if(!is_cuda_available()) {
            throw std::runtime_error("CUDA não está disponível no sistema");
        }

        if(device_id < 0 || device_id >= cuda_device_count()) {
            throw std::runtime_error("ID do dispositivo inválido: " + std::to_string(device_id));
        }

        cudaDeviceProp prop;
        cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
        if(error != cudaSuccess) {
            throw std::runtime_error("Erro ao obter propriedades do dispositivo: " + std::string(cudaGetErrorString(error)));
        }

        size_t free_mem, total_mem;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_mem, &total_mem);

        CudaDeviceInfo info;
        info.device_id = device_id;
        info.name = std::string(prop.name);
        info.total_memory = total_mem;
        info.free_memory = free_mem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.max_threads_per_block = prop.maxThreadsPerBlock;
        info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
        info.warp_size = prop.warpSize;
        info.shared_memory_per_block = prop.sharedMemPerBlock;
        info.constant_memory = prop.totalConstMem;
        info.max_grid_size_x = prop.maxGridSize[0];
        info.max_grid_size_y = prop.maxGridSize[1];
        info.max_grid_size_z = prop.maxGridSize[2];
        info.max_block_dim_x = prop.maxThreadsDim[0];
        info.max_block_dim_y = prop.maxThreadsDim[1];
        info.max_block_dim_z = prop.maxThreadsDim[2];
        info.unified_addressing = prop.unifiedAddressing;
        info.can_map_host_memory = prop.canMapHostMemory;

        return info;
#else
        throw std::runtime_error("CUDA não está disponível - projeto compilado sem suporte CUDA");
#endif
    }

    std::vector<CudaDeviceInfo> list_cuda_devices() {
        std::vector<CudaDeviceInfo> devices;
        
#if __NVCC__
        int device_count = cuda_device_count();
        
        for(int i = 0; i < device_count; ++i) {
            try {
                devices.push_back(cuda_device_info(i));
            } catch (const std::exception& e) {
                std::cerr << "Erro ao obter informações do dispositivo " << i << ": " << e.what() << std::endl;
            }
        }
#endif
        
        return devices;
    }

    int current_cuda_device() {
#if __NVCC__
        int curr_device = -1;

        if(is_cuda_available()) {
            cudaError_t error = cudaGetDevice(&curr_device);
            if(error != cudaSuccess) curr_device = -1;
        }
        
        return curr_device;
#else
        return -1;
#endif
    }

    bool set_cuda_device(int device_id) {
#if __NVCC__
        bool available = is_cuda_device_available(device_id);

        if(available) {            
            cudaError_t error = cudaSetDevice(device_id);
            available = (error == cudaSuccess);
        }

        return available;
#else
        return false;
#endif
    }

    bool is_cuda_device_available(int device_id) {
        return (device_id >= 0 && device_id < cuda_device_count());
    }

    std::pair<size_t, size_t> cuda_memory_info(int device_id) {
#if __NVCC__
        if(!is_cuda_available()) {
            throw std::runtime_error("CUDA não está disponível no sistema");
        }

        if(device_id == -1) {
            device_id = current_cuda_device();
            if(device_id == -1) throw std::runtime_error("Não foi possível obter o dispositivo CUDA atual");
        }

        if(!is_cuda_device_available(device_id)) {
            throw std::runtime_error("ID do dispositivo inválido: " + std::to_string(device_id));
        }

        const int original_device = current_cuda_device();
        cudaSetDevice(device_id);
        
        size_t free_mem, total_mem;
        cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
        
        if(original_device != -1) {
            cudaSetDevice(original_device);
        }
        
        if(error != cudaSuccess) {
            throw std::runtime_error("Erro ao obter informações de memória: " + std::string(cudaGetErrorString(error)));
        }

        return std::make_pair(free_mem, total_mem);
#else
        throw std::runtime_error("CUDA não está disponível - projeto compilado sem suporte CUDA");
#endif
    }

    bool supports_compute_capability(int device_id, int major_version, int minor_version) {
#if __NVCC__
        bool response = false;

        try {
            CudaDeviceInfo info = cuda_device_info(device_id);
            
            if (info.compute_capability_major > major_version) {
                response = true;
            } else if (info.compute_capability_major == major_version) {
                response = info.compute_capability_minor >= minor_version;
            } else {
                response = false;
            }
        } catch (const std::exception&) {
            response = false;
        }

        return false;
#else
        return false;
#endif
    }

    std::string cuda_driver_version() {
#if __NVCC__
        std::stringstream ss;

        if(is_cuda_available()) {
            int driver_version;
            cudaError_t error = cudaDriverGetVersion(&driver_version);
            
            if(error == cudaSuccess) {
                const int major = driver_version / 1000;
                const int minor = (driver_version % 1000) / 10;
                
                ss << major << "." << minor;
            }
        }

        return ss.str();
#else
        return "N/A (CUDA não compilado)";
#endif
    }

    std::string cuda_runtime_version() {
#if __NVCC__
        std::stringstream ss;

        if(is_cuda_available()) {
            int runtime_version;
            cudaError_t error = cudaRuntimeGetVersion(&runtime_version);
            
            if(error == cudaSuccess) {
                const int major = runtime_version / 1000;
                const int minor = (runtime_version % 1000) / 10;
                
                ss << major << "." << minor;
            }
        }
        
        return ss.str();
#else
        return "N/A (CUDA não compilado)";
#endif
    }

}
