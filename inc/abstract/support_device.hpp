#ifndef SUPPORT_DEVICE_HPP
#define SUPPORT_DEVICE_HPP

#include "../types/device.hpp"

class SupportDevice {
    protected:
        Device current_device = Device::CPU;

    public:

        virtual void to(Device target_device) {
            current_device = target_device;
        };

        inline bool is_cuda() const {
            return current_device == Device::GPU;
        }

        inline Device device() const {
            return current_device;
        }
};

#endif