#ifndef SHAPEABLE_HPP
#define SHAPEABLE_HPP

#include "../tensor/shape.hpp"

class Shapeable {
    protected:
        Shape sh;

    public:
        // Constructors

        template <Integral... Dims>
        Shapeable(Dims... dims);

        Shapeable(Shape shape);

        Shapeable(const Shapeable& other);

        Shapeable(Shapeable&& other) noexcept;

        // Destructor

        virtual ~Shapeable() = default;

        // Accessors

        inline int ndim() const;
        inline size_t length() const;
        inline bool is_scalar() const;
        inline Shape shape() const;
        inline int shape(int i) const;

        bool can_broadcast(const Shapeable& other) const;
};

#include "../../src/abstract/shapeable.tpp"

#endif
