#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <utility>
#include <vector>
#include <cmath>

#include "abstract/loss_function.hpp"
#include "abstract/activation.hpp"
#include "abstract/layer.hpp"

#include "layer/sequential.hpp"
#include "data/batch_loader.hpp"

#include "tensor/tensor.hpp"

#include "abstract/metric_collection.hpp"

template <Numeric T> class NeuralNetwork {
    private:
        Sequential<T>* model = nullptr;
        MetricCollection<T>* metrics = nullptr;
        LossFunction<T>* loss_function = nullptr;
        
    public:
        NeuralNetwork(
            Sequential<T>* model, 
            LossFunction<T>* loss_function, 
            Optimizer<T>* optim, 
            MetricCollection<T>* metrics = nullptr
        );

        ~NeuralNetwork();

        Tensor<T> forward(const Tensor<T>& input);
        
        void backward(const Tensor<T>& targets, const Tensor<T>& predictions);
        
        void train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size = 32);
        
        void evaluate(const Tensor<T>& X, const Tensor<T>& y);

        friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<T>& nn) {
            os << "Summary of Neural Network:" << std::endl;
            os << *nn.model << std::endl;
            os << "Loss Function: " << *nn.loss_function << std::endl;
            return os;
        }
};

#include "../src/neural_network.tpp"

#endif
