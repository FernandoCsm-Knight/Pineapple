#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP

#include "../inc/neural_network.hpp"

// Constructor

template <Numeric T>
NeuralNetwork<T>::NeuralNetwork(
    Sequential<T>* model, 
    LossFunction<T>* loss_function, 
    Optimizer<T>* optim, 
    MetricCollection<T>* metrics
): model(model), metrics(metrics), loss_function(loss_function) {

    if(model == nullptr) {
        throw std::invalid_argument("Model cannot be null");
    }

    if(loss_function == nullptr) {
        throw std::invalid_argument("Loss function cannot be null");
    }

    if(optim == nullptr) {
        throw std::invalid_argument("Optimizer cannot be null");
    }

    if(model->empty()) {
        throw std::invalid_argument("Model cannot be empty");
    }

    model->set_optimizer(optim);
}

// Destructor

template <Numeric T>
NeuralNetwork<T>::~NeuralNetwork() {
    delete model;
    if(metrics) delete metrics;
    delete loss_function;
}

// Methods

template <Numeric T>
Tensor<T> NeuralNetwork<T>::forward(const Tensor<T>& input) {
    return model->forward(input);
}

template <Numeric T>
Tensor<T> NeuralNetwork<T>::backward(const Tensor<T>& grad_output) {
    return model->backward(grad_output);
}

template <Numeric T>
void NeuralNetwork<T>::train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size) {
    BatchLoader<T> loader(X, y, batch_size, true);

    model->train();

    for(int epoch = 0; epoch < epochs; ++epoch) {
        T total_loss = 0;

        if(metrics) metrics->reset();
        
        for(const auto& [batch_X, batch_y] : loader) {
            Tensor<T> predictions = forward(batch_X);
            
            T loss = loss_function->compute(predictions, batch_y);
            total_loss += loss;
            
            if(metrics) metrics->update(predictions, batch_y);
            
            backward(loss_function->gradient(predictions, batch_y));
        }
        
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Loss: " << (total_loss/loader.num_batches());
        
            if(metrics) {
                std::set<std::string> all_metrics = metrics->all_metrics();
                
                for(const std::string& metric_name : all_metrics) {
                    const float metric_value = metrics->compute(metric_name);
                    std::cout << ", " << metric_name << ": " << metric_value;
                }
            }

            std::cout << std::endl;
        }
    }
}

template <Numeric T>
void NeuralNetwork<T>::evaluate(const Tensor<T>& X, const Tensor<T>& y) {
    model->eval();
    if(metrics) {
        metrics->reset();
        metrics->update(forward(X), y.squeeze());
    }
}

#endif