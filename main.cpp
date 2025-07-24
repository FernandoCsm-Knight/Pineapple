#include "pineapple.hpp"

#include <chrono>
#include <iostream>

#define SIZE 30

void reset(Tensor<float>& tensor, Tensor<float>& other) {
    for(size_t i = 0; i < SIZE; ++i) {
        tensor(i, i) = 1;
    }
    
    for(size_t i = 0; i < other.length(); ++i) {
        other[i] = static_cast<float>(i);
    }
}

int main() {
    Tensor<float> tensor(SIZE, SIZE);
    Tensor<float> other(SIZE, SIZE);
    reset(tensor, other);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 1000; ++i) {
        other = tensor.dot(other);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    reset(tensor, other);
    tensor.to(Device::GPU);
    other.to(Device::GPU);

    start = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < 1000; ++i) {
        other = tensor.dot(other);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time on GPU: " << duration.count() << " ms" << std::endl;

    return 0;
}