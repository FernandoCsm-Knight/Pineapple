#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP

#include "../inc/neural_network.hpp"

// Construtor

template <Numeric T>
NeuralNetwork<T>::NeuralNetwork(Sequential<T>* model, LossFunction<T>* loss_function)
    : model(model), loss_function(loss_function) {

    if(model == nullptr) {
        throw std::invalid_argument("Model cannot be null");
    }

    if(loss_function == nullptr) {
        throw std::invalid_argument("Loss function cannot be null");
    }

    if(model->empty()) {
        throw std::invalid_argument("Model cannot be empty");
    }
}

// Destructor

template <Numeric T>
NeuralNetwork<T>::~NeuralNetwork() {
    delete model;
    delete loss_function;
}

// Methods

template <Numeric T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    return model->forward(input);
}

template <Numeric T>
void NeuralNetwork<T>::backward(const Tensor<T>& targets, const Tensor<T>& predictions) {
    Tensor<T> loss_grad = loss_function->gradient(predictions, targets);
    model->backward(loss_grad);
}

template <Numeric T>
void NeuralNetwork<T>::train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size) {
    BatchLoader<T> loader(X, y, batch_size, true);
    
    for(int epoch = 0; epoch < epochs; ++epoch) {
        T total_loss = 0;
        
        for(const auto& [batch_X, batch_y] : loader) {
            Tensor<T> predictions = forward(batch_X);

            T loss = loss_function->compute(predictions, batch_y);
            total_loss += loss;
            
            backward(batch_y, predictions);
        }
        
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Loss: " << (total_loss/loader.num_batches()) << std::endl;
        }
    }
}

template <Numeric T>
T NeuralNetwork<T>::evaluate(const Tensor<T>& X, const Tensor<T>& y) {
    Tensor<T> predictions = forward(X).argmax(1);
    Tensor<bool> correct = predictions == y;

    T sum = 0;
    for(const bool& el : correct) {
        if(el) sum += 1;
    }

    return sum / correct.length();
}

#endif