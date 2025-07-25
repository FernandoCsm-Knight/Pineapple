#include <pineapple/pineapple.hpp>

#include <chrono>
#include <iostream>

#define SIZE 1000

void reset(Tensor<float>& tensor, Tensor<float>& other) {
    for(size_t i = 0; i < SIZE; ++i) {
        tensor(i, i) = 1;
    }
    
    for(size_t i = 0; i < other.length(); ++i) {
        other[i] = static_cast<float>(i);
    }
}

int main() {
    omp_set_num_threads(16);

    Tensor<float> tensor(SIZE, SIZE);
    Tensor<float> other(SIZE, SIZE);
    reset(tensor, other);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 10; ++i) {
        other = tensor.dot(other);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    if(pineapple::is_cuda_available()) {
        reset(tensor, other);
        tensor.to(Device::GPU);
        other.to(Device::GPU);
    
        start = std::chrono::high_resolution_clock::now();
    
        for(int i = 0; i < 10; ++i) {
            other = tensor.dot(other);
        }
    
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Execution time on GPU: " << duration.count() << " ms" << std::endl;
    } else {
        std::cout << "No GPU available." << std::endl;
    }

    return 0;
}