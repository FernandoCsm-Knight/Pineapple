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
void NeuralNetwork<T>::train(const Tensor<T>& X, const Tensor<T>& y, int epochs, int batch_size, float validation) {
    Partition<T> partition(X, y, true);

    auto [train_data, train_target, validation_data, validation_target] = partition.stratified_split(y, validation);

    BatchLoader<T> train_loader(train_data, train_target, batch_size, true);
    BatchLoader<T> validation_loader(validation_data, validation_target, batch_size, false);

    model->train();

    for(int epoch = 0; epoch < epochs; ++epoch) {
        T total_loss = 0;

        if(metrics) metrics->reset();
        
        for(const auto& [batch_X, batch_y] : train_loader) {
            Tensor<T> predictions = forward(batch_X);
            
            const T loss = loss_function->compute(predictions, batch_y);
            total_loss += loss;
            
            if(metrics) metrics->update(predictions, batch_y);
            
            backward(loss_function->gradient(predictions, batch_y));
        }
        
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Train Loss: " << (total_loss / train_loader.num_batches());
        
            if(metrics) {
                std::set<std::string> all_metrics = metrics->all_metrics();
                
                for(const std::string& metric_name : all_metrics) {
                    const float metric_value = metrics->compute(metric_name);
                    std::cout << ", " << metric_name << ": " << metric_value;
                }
            }

            std::cout << std::endl;
        }

        if(validation > 0.0f) {
            total_loss = 0;

            for(const auto& [val_X, val_y] : validation_loader) {
                Tensor<T> val_predictions = forward(val_X);

                const T val_loss = loss_function->compute(val_predictions, val_y);
                total_loss += val_loss;
                
                if(metrics) metrics->update(val_predictions, val_y);   
            }

            if((epoch + 1) % 10 == 0 || epoch == 0) {
                std::cout << "Época " << (epoch + 1) << ", Validation Loss: " << (total_loss / validation_loader.num_batches());
                
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