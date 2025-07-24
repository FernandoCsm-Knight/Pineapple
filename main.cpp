#include "pineapple.hpp"

int main() {
    Tensor<float> tensor(Shape{3, 3}, {
        1, 2, 3,
        4, 5, 6, 
        7, 8, 9
    });

    Tensor<float> other(Shape{3, 3}, {
        1, 1, 1,
        2, 2, 2,
        3, 3, 3
    });

    std::cout << "=== CPU Operations ===" << std::endl;

    std::cout << "Sum:" << std::endl;
    std::cout << (tensor + other) << std::endl;

    std::cout << "Sub:" << std::endl;
    std::cout << (tensor - other) << std::endl;

    std::cout << "Mul:" << std::endl;
    std::cout << (tensor * other) << std::endl;

    std::cout << "Div:" << std::endl;
    std::cout << (tensor / other) << std::endl;

    std::cout << "Matmul:" << std::endl;
    std::cout << (tensor.dot(other)) << std::endl;

#ifdef PINEAPPLE_CUDA_ENABLED
    std::cout << "\n=== GPU Operations ===" << std::endl;
    
    // Move tensors to GPU
    tensor.to(Device::GPU);
    other.to(Device::GPU);

    std::cout << "Tensors moved to GPU" << std::endl;
    std::cout << "Tensor device: " << (tensor.is_cuda() ? "GPU" : "CPU") << std::endl;
    std::cout << "Other device: " << (other.is_cuda() ? "GPU" : "CPU") << std::endl;
    
    auto gpu_sum = tensor + other;
    auto gpu_sub = tensor - other; 
    auto gpu_mul = tensor * other;
    auto gpu_div = tensor / other;
    auto gpu_matmul = tensor.dot(other);
    auto gpu_and = tensor && other;
    auto gpu_or = tensor || other;
    auto gpu_not = !tensor;
    auto gpu_eq = tensor == other;
    auto gpu_neq = tensor != other;
    auto gpu_gt = tensor > other;
    auto gpu_lt = tensor < other;
    auto gpu_ge = tensor >= other;
    auto gpu_le = tensor <= other;

    // Move results back to CPU for printing
    gpu_sum.to(Device::CPU);
    gpu_sub.to(Device::CPU);
    gpu_mul.to(Device::CPU);
    gpu_div.to(Device::CPU);
    gpu_matmul.to(Device::CPU);
    gpu_and.to(Device::CPU);
    gpu_or.to(Device::CPU);
    gpu_not.to(Device::CPU);
    gpu_eq.to(Device::CPU);
    gpu_neq.to(Device::CPU);
    gpu_gt.to(Device::CPU);
    gpu_lt.to(Device::CPU);
    gpu_ge.to(Device::CPU);
    gpu_le.to(Device::CPU);

    std::cout << "GPU Sum:" << std::endl;
    std::cout << gpu_sum << std::endl;

    std::cout << "GPU Sub:" << std::endl;
    std::cout << gpu_sub << std::endl;

    std::cout << "GPU Mul:" << std::endl;
    std::cout << gpu_mul << std::endl;

    std::cout << "GPU Div:" << std::endl;
    std::cout << gpu_div << std::endl;

    std::cout << "GPU Matmul:" << std::endl;
    std::cout << gpu_matmul << std::endl;

    std::cout << "GPU And:" << std::endl;
    std::cout << gpu_and << std::endl;

    std::cout << "GPU Or:" << std::endl;
    std::cout << gpu_or << std::endl;

    std::cout << "GPU Not:" << std::endl;
    std::cout << gpu_not << std::endl;

    std::cout << "GPU Eq:" << std::endl;
    std::cout << gpu_eq << std::endl;

    std::cout << "GPU Neq:" << std::endl;
    std::cout << gpu_neq << std::endl;

    std::cout << "GPU Gt:" << std::endl;
    std::cout << gpu_gt << std::endl;

    std::cout << "GPU Lt:" << std::endl;
    std::cout << gpu_lt << std::endl;

    std::cout << "GPU Ge:" << std::endl;
    std::cout << gpu_ge << std::endl;

    std::cout << "GPU Le:" << std::endl;
    std::cout << gpu_le << std::endl;
#else
    std::cout << "\nCUDA support not enabled in this build." << std::endl;
#endif

    return 0;
}