#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include <random>
#include <fstream>
#include <iostream>
#include <functional>

#include "tensor/tensor.hpp"

namespace pineapple {

    template <Numeric T>
    void save_to_csv(const std::string& filename, const Tensor<T>& features, const Tensor<T>& labels) {
        std::ofstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        // Write header
        file << "x,y,class" << std::endl;
        
        // Write data rows
        for (int i = 0; i < features.shape(0); i++) {
            // Write features (x,y)
            file << features(i, 0).value() << "," << features(i, 1).value() << ",";
            
            // Write class label (single value)
            file << labels[i];
            file << std::endl;
        }
        
        file.close();
    }

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_ellipses(
        int num_samples,
        int num_classes,
        float center_x = 0.0f, 
        float center_y = 0.0f, 
        float noise_max = 0.3f, 
        float a_max = 100.0f, 
        float b_max = 50.0f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_hyperbolas(
        int num_samples,
        int num_classes,
        float center_x = 0.0f,
        float center_y = 0.0f,
        float noise_max = 20.0f,
        float a_max = 100.0f,
        float b_max = 50.0f,
        float range = 100.0f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_bubbles(
        int num_samples,
        int num_classes,
        float spread_factor = 1.0f,
        float r = 1.0f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_moons(
        int num_samples,
        int num_classes,
        float noise_max = 0.1f,
        float r = 1.0f,
        float gap_h = 1.0f,
        float gap_v = 0.0f,
        float sigma = 5.0f,
        float center_x = 0.0f,
        float center_y = 0.0f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_zebra(
        int num_samples,
        int num_classes,
        float center_x = 0.0f,
        float center_y = 0.0f,
        float range = 100.0f,
        float gap = 30.0f,
        float noise_max = 10.0f,
        float theta = 1.0f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> regression_dataset(
        int num_samples, 
        int num_features,
        float lower_bound = 0.0f,
        float upper_bound = 100.0f,
        float noise_max = 0.1f
    );

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> spread_function(
        int num_samples,
        int num_features,
        const std::function<T(const Tensor<T>& coef)>& func,
        float lower_bound = 0.0f,
        float upper_bound = 100.0f,
        float noise_max = 0.1f
    );

}

#include "../src/generator.tpp"

#endif