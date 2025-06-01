#ifndef NUMERIC_HPP
#define NUMERIC_HPP

#include <type_traits>
#include <concepts>

template <typename T> concept Numeric = std::is_arithmetic_v<T>;
template <typename T> concept Integral = std::is_integral_v<T>;
template <typename T> concept Floating = std::is_floating_point_v<T>;

template <Numeric T> class Iterator {
    public:
        using value_type = T;
        using reference = T&;
        using pointer = T*;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
    
    private:
        pointer ptr;

    public:
        explicit Iterator(pointer ptr = nullptr);
        
        reference operator*() const;
        pointer operator->() const;
        reference operator[](difference_type index) const;

        difference_type operator-(const Iterator& other) const;
        Iterator operator-(difference_type offset) const;
        Iterator operator+(difference_type offset) const;

        Iterator& operator++();
        Iterator operator++(int);
        Iterator& operator--();
        Iterator operator--(int);

        Iterator& operator+=(difference_type offset);
        Iterator& operator-=(difference_type offset);

        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;
        bool operator<(const Iterator& other) const;
        bool operator>(const Iterator& other) const;
        bool operator<=(const Iterator& other) const;
        bool operator>=(const Iterator& other) const;
};

#include "../../src/types/numeric.tpp"

#endif 