#ifndef SHAPABLE_TPP
#define SHAPABLE_TPP

#include "../../inc/abstract/shapeable.hpp"

template <Integral... Dims>
Shapeable::Shapeable(Dims... dims) {
    sh = Shape(dims...);
}

Shapeable::Shapeable(Shape shape) {
    sh = shape;
}

Shapeable::Shapeable(const Shapeable& other) {
    sh = other.sh;
}

Shapeable::Shapeable(Shapeable&& other) noexcept {
    sh = std::move(other.sh);
}

int Shapeable::ndim() const {
    return sh.ndim();
}

size_t Shapeable::length() const {
    return sh.length();
}

bool Shapeable::is_scalar() const {
    return sh.ndim() == 0;
}

Shape Shapeable::shape() const {
    return sh;
}

int Shapeable::shape(int i) const {
    return sh[i];
}

bool Shapeable::can_broadcast(const Shapeable& other) const {
    return sh.can_broadcast(other.sh);
}

#endif
