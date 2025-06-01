#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <initializer_list>

#include "../abstract/layer.hpp"
#include "../tensor/tensor.hpp"

template <Numeric T> class Sequential: public Layer<T> {
    private:
        Layer<T>** layers;
        int size;

    public:
        Sequential(std::initializer_list<Layer<T>*> layer_list);
        ~Sequential();

        Tensor<T> forward(const Tensor<T>& input) override;
        Tensor<T> backward(const Tensor<T>& grad_output) override;

        Layer<T>* first() const;
        Layer<T>* last() const;

        Layer<T>* operator[](int index) const;

        bool length() const;
        bool empty() const;

        Layer<T>** begin();
        Layer<T>** end();
        
        const Layer<T>** begin() const;
        const Layer<T>** end() const;

        friend std::ostream& operator<<(std::ostream& os, const Sequential<T>& sequential) {
            os << "Sequential(" << std::endl;
            for (int i = 0; i < sequential.size; ++i) {
                os << *sequential.layers[i];
                if (i < sequential.size - 1) {
                    os << ", " << std::endl;
                }
            }
            os << std::endl << ")";
            return os;
        } 
};

#include "../../src/layer/sequential.tpp"

#endif