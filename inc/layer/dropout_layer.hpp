#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "../types/numeric.hpp"
#include "../tensor/tensor.hpp"
#include "../abstract/layer.hpp"

#include <random>

template <Numeric T> class Dropout: public Layer<T> {
    private:
        T dropout_fraction;
        Tensor<bool> mask;

    public:
        Dropout(T dropout_rate = 0.5);

        void to(Device target_device) override;
        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        bool is_activation() const override;
        bool is_optimization() const override;

        T dropout_rate() const;

        friend std::ostream& operator<<(std::ostream& os, const Dropout<T>& layer) {
            os << "DropoutLayer(dropout_rate=" << layer.dropout_fraction << ")";
            return os;
        }
};

#include "../../src/layer/dropout_layer.tpp"

#endif
