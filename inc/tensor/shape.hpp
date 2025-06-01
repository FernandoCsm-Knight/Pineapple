#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <initializer_list>
#include <iostream>

#include "../types/numeric.hpp"

class Shape {
    private:
        int* buff = nullptr;
        int dimensions;
        size_t len;

    public:
        // Constructors

        template <Integral... Dims>
        Shape(Dims... dims);

        Shape(std::initializer_list<int> dims);

        Shape(const Shape& other);
        Shape(Shape&& other) noexcept;

        // Destructor

        ~Shape();

        // Assignment operators

        Shape& operator=(const Shape& other);
        Shape& operator=(Shape&& other) noexcept;

        // Modifiers

        void add_dimension(int dim);
        void insert_dimension(int index, int dim);
        void pop_dimension();
        void remove_dimension(int index);
        
        void transpose();
        void resize_dimension(int index, int new_size);
        void reshape(std::initializer_list<int> new_dims);
        void concatenate(const Shape& other);
        void squeeze();
        void unsqueeze(int index);

        // Accessors
        
        inline int operator[](int i) const;
        inline int& operator[](int i);
        
        inline int ndim() const;
        inline size_t length() const;
        inline bool is_scalar() const;

        // Boolean operators

        bool can_broadcast(const Shape& other) const;

        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;

        // Iterators

        int* begin();
        int* end();

        const int* begin() const;
        const int* end() const;

        // Formatted output

        friend std::ostream& operator<<(std::ostream& os, const Shape& sh) {
            os << "(";
            for(int i = 0; i < sh.ndim(); ++i) {
                os << sh[i];
                if(i < sh.ndim() - 1) {
                    os << ", ";
                }
            }
            os << ")";
            
            return os;
        }
};

#include "../../src/tensor/shape.tpp"

#endif 