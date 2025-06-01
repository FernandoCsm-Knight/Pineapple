#ifndef BATCH_NORMALIZATION_HPP
#define BATCH_NORMALIZATION_HPP

#include "../tensor/tensor.hpp"
#include "../types/numeric.hpp"
#include "../abstract/layer.hpp"

template <Numeric T> class BatchNormalization: public Layer<T> {
    private:
        Tensor<T> gamma;
        Tensor<T> beta;
        
        Tensor<T> running_mean;
        Tensor<T> running_var;
        
        T epsilon;
        T momentum;
        bool is_training;
        
        Shape input_shape;
        Tensor<T> input;    
        Tensor<T> normalized;
        Tensor<T> std_dev;
        Tensor<T> var;
        Tensor<T> mean;
        
    public:
        BatchNormalization(int num_features, T epsilon = 1e-5, T momentum = 0.1);
        ~BatchNormalization() = default;

        void train();
        void eval();

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        Tensor<T> get_gamma() const { return gamma; }
        Tensor<T> get_beta() const { return beta; }
        
        void update_gamma(const Tensor<T>& dg) { gamma -= dg; }
        void update_beta(const Tensor<T>& db) { beta -= db; }

        friend std::ostream& operator<<(std::ostream& os, const BatchNormalization<T>& layer) {
            os << "BatchNorm(num_features=" << layer.gamma.shape(0) 
               << ", epsilon=" << layer.epsilon 
               << ", momentum=" << layer.momentum << ")";
            return os;
        }
};

#include "../../src/layer/batch_normalization.tpp"

#endif