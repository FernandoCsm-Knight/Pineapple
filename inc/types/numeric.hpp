#ifndef NUMERIC_HPP
#define NUMERIC_HPP

#include <type_traits>
#include <concepts>

template <typename T> concept Numeric = std::is_arithmetic_v<T>;
template <typename T> concept Integral = std::is_integral_v<T>;
template <typename T> concept Floating = std::is_floating_point_v<T>;

#endif 