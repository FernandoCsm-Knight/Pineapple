#ifndef METRIC_HPP
#define METRIC_HPP

#include <cmath>
#include <limits>
#include <string>

#include "support_device.hpp"
#include "../tensor/tensor.hpp"

template <Numeric T> class Metric: public SupportDevice {
    protected:
        Average average = Average::macro;

    public:
        Metric(Average avg = Average::macro): average(avg) {}
        virtual ~Metric() = default;
        
        inline Average average_type() const { return average; }
        virtual void to(Device target_device) override {};

        virtual std::string name() const = 0;
        virtual float compute(int TP, int TN, int FP, int FN) const = 0;
        virtual float compute(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;
};

#endif 
