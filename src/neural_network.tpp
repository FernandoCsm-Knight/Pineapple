#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP

#include "../inc/neural_network.hpp"
#include <chrono>

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
void NeuralNetwork<T>::to(Device target_device) {
    model->to(target_device);
    if(metrics) metrics->to(target_device);
    loss_function->to(target_device);
    this->current_device = target_device;
}

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
        
        // Timing variables
        double total_data_transfer_time = 0.0;
        double total_forward_time = 0.0;
        double total_loss_time = 0.0;
        double total_backward_time = 0.0;
        
        for(auto [batch_X, batch_y] : train_loader) {
            // Measure data transfer time
            auto start_transfer = std::chrono::high_resolution_clock::now();
            batch_X.to(this->device());
            batch_y.to(this->device());
            auto end_transfer = std::chrono::high_resolution_clock::now();
            auto transfer_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_transfer - start_transfer);
            total_data_transfer_time += transfer_duration.count() / 1000.0; // Convert to milliseconds

            // Measure forward pass time
            auto start_forward = std::chrono::high_resolution_clock::now();
            Tensor<T> predictions = forward(batch_X);
            auto end_forward = std::chrono::high_resolution_clock::now();
            auto forward_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_forward - start_forward);
            total_forward_time += forward_duration.count() / 1000.0; // Convert to milliseconds
            
            // Measure loss computation time
            auto start_loss = std::chrono::high_resolution_clock::now();
            const T loss = loss_function->compute(predictions, batch_y);
            total_loss += loss;
            auto end_loss = std::chrono::high_resolution_clock::now();
            auto loss_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_loss - start_loss);
            total_loss_time += loss_duration.count() / 1000.0; // Convert to milliseconds
            
            if(metrics) metrics->update(predictions, batch_y);
            
            // Measure backward pass time
            auto start_backward = std::chrono::high_resolution_clock::now();
            backward(loss_function->gradient(predictions, batch_y));
            auto end_backward = std::chrono::high_resolution_clock::now();
            auto backward_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_backward - start_backward);
            total_backward_time += backward_duration.count() / 1000.0; // Convert to milliseconds
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            std::cout << "Época " << (epoch + 1) << ", Train Loss: " << (total_loss / train_loader.num_batches());
        
            if(metrics) {
                std::set<std::string> all_metrics = metrics->all_metrics();
                
                for(const std::string& metric_name : all_metrics) {
                    const float metric_value = metrics->compute(metric_name);
                    std::cout << ", " << metric_name << ": " << metric_value;
                }
                metrics->to(this->device());
            }

            // Print timing information
            std::cout << std::endl;
            std::cout << "  Timing per epoch - Data Transfer: " << total_data_transfer_time << "ms, "
                      << "Forward: " << total_forward_time << "ms, "
                      << "Loss: " << total_loss_time << "ms, "
                      << "Backward: " << total_backward_time << "ms" << std::endl;
        }

        if(validation > 0.0f) {
            total_loss = 0;

            for(auto [val_X, val_y] : validation_loader) {
                val_X.to(this->device());
                val_y.to(this->device());

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

                    metrics->to(this->device());
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