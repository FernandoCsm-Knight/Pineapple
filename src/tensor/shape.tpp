#ifndef SHAPE_TPP
#define SHAPE_TPP

#include "../../inc/tensor/shape.hpp"

// Constructors

template <Integral... Dims>
Shape::Shape(Dims... dims) {
    const int dims_array[] = {static_cast<int>(dims)...};
    const int num_dims = sizeof...(dims);
    
    len = 1;
    if(num_dims > 1 && dims_array[0] == 0) {
        throw std::invalid_argument("First dimension cannot be zero if there are multiple dimensions");
    } 

    if(num_dims == 0 || dims_array[0] == 0) {
        dimensions = 0;
        buff = new int[1]{ 0 };
    } else {
        dimensions = num_dims;
        buff = new int[dimensions];
        
        for(int i = 0; i < dimensions; ++i) {
            if(dims_array[i] <= 0) {
                throw std::invalid_argument("Dimension sizes must be positive");
            }

            buff[i] = dims_array[i];
            len *= dims_array[i];
        }
    }
}

Shape::Shape(std::initializer_list<int> dims) {
    const int num_dims = dims.size();
    
    len = 1;
    if(num_dims > 1 && *(dims.begin()) == 0) {
        throw std::invalid_argument("First dimension cannot be zero if there are multiple dimensions");
    }

    if(num_dims == 0 || *(dims.begin()) == 0) {
        dimensions = 0;
        buff = new int[1]{ 0 };
    } else {
        dimensions = num_dims;
        buff = new int[dimensions];
        
        int i = 0;
        for(int el : dims) {
            if(el <= 0) throw std::invalid_argument("Dimension sizes must be positive");
            
            buff[i] = el;
            len *= el;
            ++i;
        }
    }
}

Shape::Shape(const Shape& other) {
    len = other.len;
    dimensions = other.dimensions;

    buff = new int[dimensions];
    for(int i = 0; i < dimensions; ++i) {
        buff[i] = other.buff[i];
    }
}

Shape::Shape(Shape&& other) noexcept {
    len = other.len;
    buff = other.buff;
    dimensions = other.dimensions;
    
    other.len = 0;
    other.dimensions = 0;
    other.buff = nullptr;
}

// Destructor

Shape::~Shape() {
    delete[] buff;
}

// Assignment operators

Shape& Shape::operator=(const Shape& other) {
    if(this != &other) {
        delete[] buff;
        len = other.len;
        dimensions = other.dimensions;

        buff = new int[dimensions];
        for(int i = 0; i < dimensions; ++i) {
            buff[i] = other.buff[i];
        }
    }

    return *this;
}

Shape& Shape::operator=(Shape&& other) noexcept {
    if (this != &other) {
        delete[] buff;
        len = other.len;
        buff = other.buff;
        dimensions = other.dimensions;

        other.len = 0;
        other.dimensions = 0;
        other.buff = nullptr;
    }

    return *this;
}

// Modifiers

void Shape::add_dimension(int dim) {
    this->insert_dimension(dimensions, dim);
}

void Shape::insert_dimension(int index, int dim) {
    if (index < 0 || index > dimensions) {
        throw std::out_of_range("Insert position out of range");
    }
    
    if (dim <= 0) {
        throw std::invalid_argument("Dimension size must be positive");
    }
    
    int* new_buff = new int[dimensions + 1];
    for (int i = 0; i < index; ++i) {
        new_buff[i] = buff[i];
    }
    
    new_buff[index] = dim;
    for (int i = index; i < dimensions; ++i) {
        new_buff[i + 1] = buff[i];
    }
    
    delete[] buff;
    buff = new_buff;
    dimensions++;
    len *= dim;
}

void Shape::pop_dimension() {
    this->remove_dimension(dimensions - 1);
}

void Shape::remove_dimension(int index) {
    if (index < 0 || index >= dimensions) {
        throw std::out_of_range("Dimension index out of range");
    }
    
    if (dimensions <= 1) {
        delete[] buff;
        dimensions = 0;
        buff = new int[1]{ 0 };
        len = 1;
    } else {
        int* new_buff = new int[dimensions - 1];
        
        for(int i = 0, j = 0; i < dimensions; ++i) {
            if(i != index) new_buff[j++] = buff[i];
        }
        
        len /= buff[index];
        
        delete[] buff;
        buff = new_buff;
        dimensions--;
    }
}

void Shape::resize_dimension(int index, int new_dim) {
    if(index < 0 || index >= dimensions) {
        throw std::out_of_range("Dimension index out of range");
    }
    
    if(new_dim <= 0) {
        throw std::invalid_argument("New dimension size must be positive");
    }
    
    len /= buff[index];
    len *= new_dim;
    buff[index] = new_dim;
}

void Shape::transpose() {    
    int i = 0, j = dimensions - 1;
    while (i < j) {
        std::swap(buff[i], buff[j]);
        ++i;
        --j;
    }
}

void Shape::reshape(std::initializer_list<int> new_dims) {
    size_t new_len = 1;
    for(int d : new_dims) {
        if(d <= 0) throw std::invalid_argument("Dimension sizes must be positive");
        new_len *= d;
    }
    
    if (new_len != len) {
        throw std::invalid_argument("New shape must preserve total number of elements");
    }
    
    delete[] buff;
    dimensions = new_dims.size();
    buff = new int[dimensions];
    
    int i = 0;
    for (int d : new_dims) {
        buff[i++] = d;
    }
}

void Shape::concatenate(const Shape& other) {
    int* new_buff = new int[dimensions + other.dimensions];
    
    for (int i = 0; i < dimensions; ++i) {
        new_buff[i] = buff[i];
    }
    
    for (int i = 0; i < other.dimensions; ++i) {
        new_buff[dimensions + i] = other.buff[i];
    }
    
    delete[] buff;
    buff = new_buff;
    
    for (int i = 0; i < other.dimensions; ++i) {
        len *= other.buff[i];
    }
    
    dimensions += other.dimensions;
}

void Shape::squeeze() {
    if(dimensions != 0) {
        int ones = 0;
        for(int i = 0; i < dimensions; ++i) {
            if(buff[i] == 1) ones++;
        }

        len = 1;

        if(ones == dimensions) {
            delete[] buff;
            dimensions = 0;
            buff = new int[1]{ 0 };
        } else {
            int* new_buff = new int[dimensions - ones];
            int j = 0;
            for(int i = 0; i < dimensions; ++i) {
                if(buff[i] != 1) {
                    new_buff[j++] = buff[i];
                }
            }
            
            delete[] buff;
            buff = new_buff;
            dimensions -= ones;
            for(int i = 0; i < dimensions; ++i) {
                len *= buff[i];
            }
        }
    }
}

void Shape::unsqueeze(int index) {
    if(index < 0 || index > dimensions) {
        throw std::out_of_range("Insert position out of range");
    }
    
    int* new_buff = new int[dimensions + 1];
    for (int i = 0; i < index; ++i) {
        new_buff[i] = buff[i];
    }
    
    new_buff[index] = 1;
    for (int i = index; i < dimensions; ++i) {
        new_buff[i + 1] = buff[i];
    }
    
    delete[] buff;
    buff = new_buff;
    dimensions++;
}

// Accessors

int Shape::operator[](int i) const {
    return buff[i];
}

int& Shape::operator[](int i) {
    return buff[i];
}

int Shape::ndim() const {
    return dimensions;
}

size_t Shape::length() const {
    return (dimensions == 0) ? 1 : len;
}

bool Shape::is_scalar() const {
    return dimensions == 0;
}

// Boolean operators

bool Shape::can_broadcast(const Shape& other) const {
    int limit = std::min(this->ndim(), other.ndim());
    bool broadcastable = true;
    
    for(int i = 1; broadcastable && i <= limit; ++i) {
        broadcastable = buff[this->ndim() - i] == other[other.ndim() - i]
                        || buff[this->ndim() - i] == 1
                        || other[other.ndim() - i] == 1;
    }
    
    return broadcastable;
}

bool Shape::operator==(const Shape& other) const {
    bool equal = dimensions == other.dimensions && len == other.len;

    for(int i = 0; equal && i < dimensions; ++i) 
        equal = buff[i] == other.buff[i];

    return equal;
}

bool Shape::operator!=(const Shape& other) const {
    return !(*this == other);
}

// Iterators

int* Shape::begin() {
    return buff;
}

int* Shape::end() {
    return buff + dimensions;
}

const int* Shape::begin() const {
    return buff;
}

const int* Shape::end() const {
    return buff + dimensions;
}

#endif
