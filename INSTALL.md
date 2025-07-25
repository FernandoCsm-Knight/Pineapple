# Installing Pineapple Library

This document explains how to install the Pineapple library on your Linux system.

## Installation Options

The Pineapple library supports both CPU-only and CUDA-accelerated installations.

### Quick Installation (CPU-only)

Clone the repository and run the installation script:

```bash
git clone https://github.com/FernandoCsm-Knight/Pineapple.git
cd Pineapple
./install.sh
```

### CUDA Installation

For GPU acceleration with CUDA support:

```bash
git clone https://github.com/FernandoCsm-Knight/Pineapple.git
cd Pineapple
CUDA=1 ./install.sh
```

## Installation Methods

### Method 1: Installation Script (Recommended)

#### CPU-only installation:
```bash
./install.sh
```

#### CUDA-enabled installation:
```bash
CUDA=1 ./install.sh
```

#### Custom installation path:
```bash
PREFIX=/opt/pineapple ./install.sh
```

#### Advanced CUDA options:
```bash
# Specific CUDA architectures
CUDA=1 CUDA_ARCH=sm_75,sm_80,sm_86 ./install.sh

# Custom path with CUDA
CUDA=1 PREFIX=/opt/pineapple ./install.sh
```

### Method 2: Manual Installation (CPU-only)

```bash
sudo mkdir -p /usr/local/include/pineapple
sudo cp -r inc /usr/local/include/pineapple/
sudo cp -r src /usr/local/include/pineapple/
sudo cp pineapple.hpp /usr/local/include/pineapple/

sudo tee /usr/local/lib/pkgconfig/pineapple.pc > /dev/null << 'EOF'
prefix=/usr/local
exec_prefix=${prefix}
includedir=${prefix}/include

Name: pineapple
Description: Pineapple Neural Network Library
Version: 1.0.0
Cflags: -I${includedir}/pineapple -fopenmp -std=c++20
Libs: -fopenmp
EOF
```

## Prerequisites

- **C++20 Compiler**: g++ or clang++ or MSVC 2019+
- **OpenMP**: For parallelization (optional but recommended)
- **pkg-config**: To simplify usage (optional)

### Installing dependencies:

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential pkg-config
```

#### Fedora/RHEL/CentOS:
```bash
sudo dnf install gcc-c++ make pkgconfig
```

#### Arch Linux:
```bash
sudo pacman -S gcc make pkgconf
```

## Usage After Installation

### 1. Include in your code:
```cpp
#include <pineapple/pineapple.hpp>

int main() {
    Tensor<float> tensor(Shape{3, 3}, {
        1, 2, 3,
        4, 5, 6, 
        7, 8, 9
    });
    
    std::cout << tensor << std::endl;
    return 0;
}
```

### 2. Compile using pkg-config (recommended):
```bash
g++ -std=c++20 $(pkg-config --cflags pineapple) your_program.cpp $(pkg-config --libs pineapple) -o your_program
```

### 3. Or compile manually:
```bash
g++ -std=c++20 -I/usr/local/include/pineapple -fopenmp your_program.cpp -o your_program
```

### 4. Using with CMake:
```cmake
cmake_minimum_required(VERSION 3.16)
project(YourProject)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PINEAPPLE REQUIRED pineapple)

find_package(OpenMP REQUIRED)

add_executable(your_program your_program.cpp)
target_include_directories(your_program PRIVATE ${PINEAPPLE_INCLUDE_DIRS})
target_compile_options(your_program PRIVATE ${PINEAPPLE_CFLAGS_OTHER})
target_link_libraries(your_program OpenMP::OpenMP_CXX)
```

## Uninstallation

### Using script (Recommended):
```bash
./uninstall.sh
```

### Manual:
```bash
sudo rm -rf /usr/local/include/pineapple
sudo rm -f /usr/local/lib/pkgconfig/pineapple.pc
```

## Custom Installation

### Different directory:
```bash
PREFIX=/opt ./install.sh
```

### Local installation (without sudo):
```bash
PREFIX=$HOME/.local ./install.sh
```

Then add to your `.bashrc` or `.zshrc`:
```bash
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig:$PKG_CONFIG_PATH"
export CPLUS_INCLUDE_PATH="$HOME/.local/include:$CPLUS_INCLUDE_PATH"
```

## Important Notes

- **Header-Only Library**: Pineapple is a template-based library, so only headers are needed
- **C++20 Required**: The library requires C++20 to function
- **OpenMP Recommended**: For better performance in parallel operations
- **.tpp Files**: Maintained in original structure - automatically included by headers

## Documentation

After installation, documentation will be available in the headers at:
- `/usr/local/include/pineapple/` (default installation)
- `$PREFIX/include/pineapple/` (custom installation)

For more information, consult the main README.md of the project.
