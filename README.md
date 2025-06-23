# Pineapple

> ### Version 0.3.0

Pineapple is a C++20 micro-framework for AI. It supports building and evaluating dense neural networks for classification and regression, it offers CPU-level parallelism. Data handling is done by Tensor class, which provides shape broadcasting, element-wise arithmetic and boolean operations, reductions, basic linear algebra, 1-D/2-D convolutions and correlations, plus flexible indexing, slicing, transposition, and reshaping, all generically typed through templates for any numeric type. At version 0.3.0 the framework focuses on supervised classification and regression with dense layers, supplying activation and loss functions, multiple optimizers and a collection of classification metrics, but it already lays an extensible foundation for future convolutional layers, GPU support, and additional optimization algorithms (see branch dev).

## Features (version 0.3.0)

- **Dense Neural Networks**: Build and train fully connected neural networks with customizable architectures
- **CPU Parallelism**: OpenMP support for parallel computation across multiple threads
- **Tensor Operations**: Generic tensor class supporting:
  - Shape broadcasting and arithmetic operations
  - Linear algebra operations
  - 1-D/2-D convolutions and correlations
  - Flexible indexing, slicing, and reshaping
- **Activation Functions**: Multiple activation functions (ReLU, Sigmoid, Tanh, ELU, Leaky ReLU, Softmax)
- **Loss Functions**: Various loss functions for classification tasks
- **Optimizers**: Multiple optimization algorithms for training
- **Metrics**: Collection of classification metrics for model evaluation

## Installation

### Prerequisites

- C++20 compatible compiler (e.g., GCC 10+, Clang 10+, MSVC 2019+)
- OpenMP support
- Make utility

### Build Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd Pineapple
```

2. Build the project:
```bash
make build
```

3. Clean build files (if needed):
```bash
make clean
```

## Execution

**Running the Program**
```bash
make run
```

**Performance Timing**
```bash
make time
```

**Memory Leak Detection**
```bash
make valgrind
```

**OpenMP Configuration**
Set the number of threads (default is 1):
```bash
make run OMP=<threads>
```

**Complete Rebuild**
```bash
make rerun OMP=<threads>
```

## Exemples

To use Pineapple, you can use the main file `main.cpp` to create your own neural networks, load datasets, and perform training and evaluation. The framework provides a simple interface for defining models, specifying loss functions, and choosing optimizers.

As you start using Pineapple, you can refer to the provided video examples for guidance on how to implement various neural network architectures. If you don't want to use the default `main.cpp`, you will need to include the `pineapple.hpp` header in your project and link the template files at your own makefile.

> Warning: In future versions CMake will be used to install and uninstall the library at windows and linux envirounments.

### Videos 

1. [Introduction to Tensor]()
2. [Introduction to Pineapple Classification]()
3. [Introduction to Pineapple Regression]()

## References

### Books

Most algorithms and concetps implemented in Pineapple are based on the following books:

- [Data Mining: Concepts and Techniques, 4th edition](https://www.educate.elsevier.com/book/details/9780128117606)
- [Deep Learning](https://www.deeplearningbook.org/)
- [Artificial Intelligence: A Modern Approach, 4th edition](https://aima.cs.berkeley.edu/)

### Libraries

Pineapple offers conveniences inspired by both frameworks listed below, simplifying common tasks in the training workflow. However, Pineapple is an independent project and is not derived from these frameworks.

- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)

## Copyright

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg