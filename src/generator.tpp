#ifndef GENERATOR_TPP
#define GENERATOR_TPP

#include "../inc/generator.hpp"
#include <random>

namespace pineapple {

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_ellipses(
        int num_samples,
        int num_classes,
        float center_x,
        float center_y,
        float noise_max,
        float a_max,
        float b_max
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> radius(0.2f, noise_max);

        Tensor<T> data(num_samples, 2);
        Tensor<T> target(num_samples);

        for (int i = 0; i < num_samples; ++i) {
            const int class_id = i % num_classes;
            const float scale = (class_id + 1) / static_cast<float>(num_classes);

            const float theta = angle(gen);
            const float r = std::sqrt(radius(gen));
            const float x = center_x + r * (a_max * scale) * std::cos(theta);
            const float y = center_y + r * (b_max * scale) * std::sin(theta);

            data(i, 0) = x;
            data(i, 1) = y;
            target[i] = class_id;
        }

        return std::make_pair(data, target);
    }

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_hyperbolas(
        int num_samples,
        int num_classes,
        float center_x,
        float center_y,
        float noise_max,
        float a_max,
        float b_max,
        float range
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> range_dist(-range, range);
        std::uniform_real_distribution<float> noise(0.0f, noise_max);

        Tensor<T> data(num_samples, 2);
        Tensor<T> target(num_samples);

        for(int i = 0; i < num_samples; i += 2) {
            const int class_id = i % num_classes;
            const float scale = (class_id + 1) / static_cast<float>(num_classes);
            const float a_i = std::pow(a_max * scale, 2);
            const float b_i = std::pow(b_max * scale / (class_id + 1), 2);
        
            const float x = range_dist(gen);
            const float y = std::sqrt((1 + std::pow(x - center_x, 2) / b_i) * a_i) + center_y + noise(gen);

            data(i, 0) = x;
            data(i + 1, 0) = x;
            data(i, 1) = y;
            data(i + 1, 1) = -y;
            target[i] = class_id;
            target[i + 1] = class_id;
        }

        return std::make_pair(data, target);
    }

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_bubbles(
        int num_samples,
        int num_classes,
        float spread_factor,
        float r
    ) {
        if(spread_factor < 0.0f || spread_factor > 1.0f) {
            throw std::invalid_argument("spread_factor deve estar em [0,1]");
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle(0.0f, 2.0f * M_PI);
        std::uniform_real_distribution<float> radius(0.0f, 1.0f);

        const float d = 2.0f * r * spread_factor;
        const float R_center = (num_classes == 1) ? 0.0f : d / (2.0f * std::sin(M_PI / num_classes));

        Tensor<T> data(num_samples, 2);
        Tensor<T> target(num_samples);

        for(int i = 0; i < num_samples; ++i) {
            const int class_id = i % num_classes;
            const float phi_c = 2.0f * M_PI * class_id / num_classes;
            const float cx = R_center * std::cos(phi_c);
            const float cy = R_center * std::sin(phi_c);

            const float theta = angle(gen);
            const float rho = r * std::sqrt(radius(gen));
            const float x = cx + rho * std::cos(theta);
            const float y = cy + rho * std::sin(theta);

            data(i, 0) = x;
            data(i, 1) = y;
            target[i] = class_id;
        }

        return std::make_pair(data, target);
    }

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_moons(
        int num_samples,
        int num_classes,
        float noise_max,
        float r,
        float gap_h,
        float gap_v,
        float sigma,
        float center_x,
        float center_y
    ) {
        if(sigma <= 0.0f) {
            throw std::invalid_argument("sigma value must be greater than 0");
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle(0.0f, M_PI);
        std::uniform_real_distribution<float> noise(-noise_max, noise_max);

        Tensor<T> data(num_samples, 2);
        Tensor<T> target(num_samples);

        for(int i = 0; i < num_samples; i++) {
            const int class_id = i % num_classes;
            const int factor = (class_id % 2 == 0) ? 1 : -1;
            const float theta = angle(gen);

            const float x = r * std::cos(theta) - (factor * gap_h / 2) + center_x;
            const float y = r * std::sin(theta) + ((class_id / 2) * r + gap_v) / 2 + (factor * center_y);

            data(i, 0) = x + noise(gen) / sigma;
            data(i, 1) = factor * y + noise(gen);
            target[i] = class_id;
        }

        return std::make_pair(data, target);
    }

    template <Numeric T>
    std::pair<Tensor<T>, Tensor<T>> create_zebra(
        int num_samples,
        int num_classes,
        float center_x,
        float center_y,
        float range,
        float gap,
        float noise_max,
        float theta
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> range_dist(-range, range);
        std::uniform_real_distribution<float> noise(0.0f, noise_max);

        Tensor<T> data(num_samples, 2);
        Tensor<T> target(num_samples);

        for(int i = 0; i < num_samples; ++i) {
            const int class_id = i % num_classes;
            
            const float b = gap * class_id - ((num_classes - 1) * gap) / 2.0f;
            const float x = range_dist(gen);
            const float y = theta * (x - center_x) + center_y + b + noise(gen);

            data(i, 0) = x;
            data(i, 1) = y;
            target[i] = class_id;
        }

        return std::make_pair(data, target);
    }

}

#endif