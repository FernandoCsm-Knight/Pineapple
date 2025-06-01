#ifndef ELU_ACTIVATION_HPP
#define ELU_ACTIVATION_HPP

#include <cmath>

#include "../abstract/activation.hpp"

template <Numeric T> class ELU: public Activation<T> {
    private:
        T alpha;

    protected:
        Tensor<T> apply(const Tensor<T>& input) const override;
        Tensor<T> derivative(const Tensor<T>& input) const override;
    
    public:
        explicit ELU(T alpha = 1.0) : alpha(alpha) {}
        
        friend std::ostream& operator<<(std::ostream& os, const ELU<T>& elu) {
            os << "ELU(alpha=" << elu.alpha << ")";
            return os;
        }
};

#include "../../src/activation/elu.tpp"

#endif