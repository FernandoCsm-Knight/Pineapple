#ifndef ITERATOR_TPP
#define ITERATOR_TPP

#include "../../inc/types/numeric.hpp"

template <Numeric T>
Iterator<T>::Iterator(pointer ptr): ptr(ptr) {}

template <Numeric T>
typename Iterator<T>::reference Iterator<T>::operator*() const {
    return *ptr;
}

template <Numeric T>
typename Iterator<T>::pointer Iterator<T>::operator->() const {
    return ptr;
}

template <Numeric T>
typename Iterator<T>::reference Iterator<T>::operator[](difference_type index) const {
    return ptr[index];
}

template <Numeric T>
typename Iterator<T>::difference_type Iterator<T>::operator-(const Iterator& other) const {
    return ptr - other.ptr;
}

template <Numeric T>
Iterator<T> Iterator<T>::operator+(difference_type offset) const {
    return Iterator(ptr + offset);
}

template <Numeric T>
Iterator<T> Iterator<T>::operator-(difference_type offset) const {
    return Iterator(ptr - offset);
}

template <Numeric T>
Iterator<T>& Iterator<T>::operator++() {
    ++ptr;
    return *this;
}

template <Numeric T>
Iterator<T> Iterator<T>::operator++(int) {
    Iterator copy = *this;
    ++ptr;
    return copy;
}

template <Numeric T>
Iterator<T>& Iterator<T>::operator--() {
    --ptr;
    return *this;
}

template <Numeric T>
Iterator<T> Iterator<T>::operator--(int) {
    Iterator copy = *this;
    --ptr;
    return copy;
}

template <Numeric T>
Iterator<T>& Iterator<T>::operator+=(difference_type offset) {
    ptr += offset;
    return *this;
}

template <Numeric T>
Iterator<T>& Iterator<T>::operator-=(difference_type offset) {
    ptr -= offset;
    return *this;
}

template <Numeric T>
bool Iterator<T>::operator==(const Iterator& other) const { 
    return ptr == other.ptr; 
}

template <Numeric T>
bool Iterator<T>::operator!=(const Iterator& other) const { 
    return ptr != other.ptr; 
}

template <Numeric T>
bool Iterator<T>::operator<(const Iterator& other) const { 
    return ptr < other.ptr; 
}

template <Numeric T>
bool Iterator<T>::operator>(const Iterator& other) const { 
    return ptr > other.ptr; 
}

template <Numeric T>
bool Iterator<T>::operator<=(const Iterator& other) const { 
    return ptr <= other.ptr; 
}

template <Numeric T>
bool Iterator<T>::operator>=(const Iterator& other) const { 
    return ptr >= other.ptr; 
}

#endif