#!/bin/bash

# Pineapple library installation script
# Author: Fernando Campos Silva Dal Maria
# Description: Installs the header-only Pineapple library on the system

set -e  # Exit on error

# Configuration
LIBRARY_NAME="pineapple"
VERSION="0.3.0"
PREFIX="${PREFIX:-/usr/local}"
INCLUDEDIR="$PREFIX/include"
LIBDIR="$PREFIX/lib"
PKGCONFIGDIR="$PREFIX/lib/pkgconfig"

# CUDA configuration
CUDA_ENABLED="${CUDA:-0}"
CUDA_ARCH="${CUDA_ARCH:-sm_50,sm_60,sm_70,sm_75,sm_80,sm_86}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        SUDO=""
    elif command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
        print_info "sudo will be required for installation"
    else
        print_error "You need root permissions or sudo installed"
        exit 1
    fi
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check g++
    if ! command -v g++ >/dev/null 2>&1; then
        print_error "g++ not found. Install C++ compiler:"
        print_info "Ubuntu/Debian: sudo apt install build-essential"
        print_info "Fedora/RHEL: sudo dnf install gcc-c++"
        print_info "Arch: sudo pacman -S gcc"
        exit 1
    fi
    
    # Check CUDA if enabled
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        print_info "CUDA support requested, checking CUDA toolkit..."
        
        if ! command -v nvcc >/dev/null 2>&1; then
            print_error "nvcc not found. CUDA toolkit is required for CUDA support."
            print_info "Install CUDA toolkit from: https://developer.nvidia.com/cuda-toolkit"
            print_info "Or disable CUDA with: CUDA=0 $0"
            exit 1
        fi
        
        # Check CUDA runtime
        if ! ldconfig -p | grep -q libcudart; then
            print_warning "CUDA runtime library may not be available in library path"
        fi
        
        print_success "CUDA toolkit found: $(nvcc --version | grep release)"
    fi
    
    # Check OpenMP
    cat > /tmp/test_omp.cpp << EOF
#include <omp.h>

int main() {
    return 0;
}
EOF

    if ! g++ -fopenmp -x c++ /tmp/test_omp.cpp -o /tmp/test_omp 2>/dev/null; then
        print_warning "OpenMP may not be available. Library will work but without parallelization"
    else
        rm -f /tmp/test_omp.cpp
        rm -f /tmp/test_omp
    fi
    
    print_success "Dependencies checked"
}

compile_cuda_libraries() {
    if [[ "$CUDA_ENABLED" != "1" ]]; then
        print_info "CUDA support disabled, skipping CUDA library compilation"
        return 0
    fi
    
    print_info "Compiling CUDA libraries..."
    
    # Create temporary build directory
    BUILD_DIR="/tmp/pineapple_build"
    mkdir -p "$BUILD_DIR"
    
    # Copy source files to build directory
    cp -r src "$BUILD_DIR/"
    cp -r inc "$BUILD_DIR/"
    
    cd "$BUILD_DIR"
    
    # Compile CUDA object files
    print_info "Compiling CUDA kernels..."
    nvcc --extended-lambda -std=c++20 \
         -gencode arch=compute_50,code=sm_50 \
         -gencode arch=compute_60,code=sm_60 \
         -gencode arch=compute_70,code=sm_70 \
         -gencode arch=compute_75,code=sm_75 \
         -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -Xcompiler -fPIC,-fopenmp \
         -Iinc \
         -c src/tensor/tensor_cuda_kernels.cu \
         -o tensor_cuda_kernels.o
    
    print_info "Compiling CUDA wrappers..."
    nvcc --extended-lambda -std=c++20 \
         -gencode arch=compute_50,code=sm_50 \
         -gencode arch=compute_60,code=sm_60 \
         -gencode arch=compute_70,code=sm_70 \
         -gencode arch=compute_75,code=sm_75 \
         -gencode arch=compute_80,code=sm_80 \
         -gencode arch=compute_86,code=sm_86 \
         -Xcompiler -fPIC,-fopenmp \
         -Iinc \
         -c src/tensor/tensor_cuda_wrappers.cu \
         -o tensor_cuda_wrappers.o
    
    # Create static library
    print_info "Creating static library..."
    ar rcs libpineapple_cuda.a tensor_cuda_kernels.o tensor_cuda_wrappers.o
    
    # Create shared library
    print_info "Creating shared library..."
    nvcc -shared -Xcompiler -fPIC \
         tensor_cuda_kernels.o tensor_cuda_wrappers.o \
         -o libpineapple_cuda.so.${VERSION} \
         -lcudart
    
    # Create version symlinks
    ln -sf libpineapple_cuda.so.${VERSION} libpineapple_cuda.so.${VERSION%.*}
    ln -sf libpineapple_cuda.so.${VERSION} libpineapple_cuda.so
    
    # Move back to original directory
    cd - > /dev/null
    
    print_success "CUDA libraries compiled successfully"
}

install_library() {
    print_info "Installing Pineapple library..."
    
    print_info "Creating installation directories..."
    $SUDO mkdir -p "$INCLUDEDIR/$LIBRARY_NAME"
    $SUDO mkdir -p "$LIBDIR"
    $SUDO mkdir -p "$PKGCONFIGDIR"
    
    print_info "Copying header files and implementations..."
    $SUDO cp -r inc "$INCLUDEDIR/$LIBRARY_NAME/"
    $SUDO cp -r src "$INCLUDEDIR/$LIBRARY_NAME/"
    $SUDO cp pineapple.hpp "$INCLUDEDIR/$LIBRARY_NAME/"
    
    # Install CUDA libraries if compiled
    if [[ "$CUDA_ENABLED" == "1" ]] && [[ -f "/tmp/pineapple_build/libpineapple_cuda.a" ]]; then
        print_info "Installing CUDA libraries..."
        $SUDO cp /tmp/pineapple_build/libpineapple_cuda.a "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple_cuda.so.${VERSION} "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple_cuda.so.${VERSION%.*} "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple_cuda.so "$LIBDIR/"
        
        # Update library cache
        if command -v ldconfig >/dev/null 2>&1; then
            $SUDO ldconfig
        fi
        
        print_success "CUDA libraries installed"
    fi
    
    if [ ! -f "$INCLUDEDIR/$LIBRARY_NAME/pineapple.hpp" ]; then
        print_error "Failed to copy main pineapple.hpp file"
        exit 1
    fi
    
    if [ ! -d "$INCLUDEDIR/$LIBRARY_NAME/inc" ]; then
        print_error "Failed to copy inc/ directory"
        exit 1
    fi
    
    if [ ! -d "$INCLUDEDIR/$LIBRARY_NAME/src" ]; then
        print_error "Failed to copy implementation files (.tpp)"
        exit 1
    fi
    
    print_info "Creating pkg-config file..."
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        # pkg-config with CUDA support
        cat > /tmp/pineapple.pc << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: $LIBRARY_NAME
Description: Pineapple Neural Network Library - Header-only C++ template library with CUDA support
Version: $VERSION
Cflags: -I\${includedir}/$LIBRARY_NAME -fopenmp -std=c++20 
Libs: -L\${libdir} -lpineapple_cuda -lcudart -fopenmp
Libs.private: -lcuda
EOF
    else
        # pkg-config without CUDA support
        cat > /tmp/pineapple.pc << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include

Name: $LIBRARY_NAME
Description: Pineapple Neural Network Library - Header-only C++ template library
Version: $VERSION
Cflags: -I\${includedir}/$LIBRARY_NAME -fopenmp -std=c++20
Libs: -fopenmp
EOF
    fi
    
    $SUDO cp /tmp/pineapple.pc "$PKGCONFIGDIR/"
    rm -f /tmp/pineapple.pc
    
    # Set appropriate permissions
    print_info "Setting permissions..."
    $SUDO chmod -R 644 "$INCLUDEDIR/$LIBRARY_NAME"
    $SUDO find "$INCLUDEDIR/$LIBRARY_NAME" -type d -exec chmod 755 {} \;
    $SUDO chmod 644 "$PKGCONFIGDIR/pineapple.pc"
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        $SUDO chmod 644 "$LIBDIR"/libpineapple_cuda.*
    fi
    
    # Clean up build directory
    rm -rf /tmp/pineapple_build
    
    print_success "Pineapple library installed successfully!"
}

test_installation() {
    print_info "Testing installation..."
    
    # Create test file
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        cat > /tmp/test_pineapple.cpp << 'EOF'
#include <pineapple/pineapple.hpp>
#include <iostream>

int main() {
    Tensor<float> tensor(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Pineapple library test: OK" << std::endl;
    std::cout << "Tensor created successfully: " << tensor.length() << " elements" << std::endl;
    
    #ifdef PINEAPPLE_CUDA_ENABLED
    std::cout << "CUDA support: ENABLED" << std::endl;
    #else
    std::cout << "CUDA support: DISABLED" << std::endl;
    #endif
    
    return 0;
}
EOF
    else
        cat > /tmp/test_pineapple.cpp << 'EOF'
#include <pineapple/pineapple.hpp>
#include <iostream>

int main() {
    Tensor<float> tensor(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Pineapple library test: OK" << std::endl;
    std::cout << "Tensor created successfully: " << tensor.length() << " elements" << std::endl;
    std::cout << "CUDA support: DISABLED" << std::endl;
    return 0;
}
EOF
    fi
    
    # Try to compile
    if pkg-config --exists pineapple 2>/dev/null; then
        if [[ "$CUDA_ENABLED" == "1" ]]; then
            # Test with nvcc for CUDA
            if nvcc --extended-lambda -std=c++20 $(pkg-config --cflags pineapple) /tmp/test_pineapple.cpp $(pkg-config --libs pineapple) -o /tmp/test_pineapple 2>/dev/null; then
                if /tmp/test_pineapple >/dev/null 2>&1; then
                    print_success "Installation test passed (with CUDA)!"
                else
                    print_warning "Compilation OK, but execution failed"
                fi
            else
                print_warning "Test compilation with nvcc failed"
            fi
        else
            # Test with g++ for CPU-only
            if g++ -std=c++20 $(pkg-config --cflags pineapple) /tmp/test_pineapple.cpp $(pkg-config --libs pineapple) -o /tmp/test_pineapple 2>/dev/null; then
                if /tmp/test_pineapple >/dev/null 2>&1; then
                    print_success "Installation test passed (CPU-only)!"
                else
                    print_warning "Compilation OK, but execution failed"
                fi
            else
                print_warning "Test compilation failed"
            fi
        fi
    else
        print_warning "pkg-config did not find the library (cache update may be needed)"
    fi
    
    # Clean test files
    rm -f /tmp/test_pineapple.cpp /tmp/test_pineapple
}

show_usage_info() {
    echo ""
    print_success "=== INSTALLATION COMPLETED ==="
    echo ""
    print_info "How to use the Pineapple library:"
    echo ""
    echo "1. Include in your C++ code:"
    echo "   #include <pineapple/pineapple.hpp>"
    echo ""
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        echo "2. Compile with CUDA support using pkg-config (recommended):"
        echo "   nvcc --extended-lambda -std=c++20 \$(pkg-config --cflags pineapple) your_file.cpp \$(pkg-config --libs pineapple) -o your_program"
        echo ""
        echo "3. Or compile manually with CUDA:"
        echo "   nvcc --extended-lambda -std=c++20 -I$INCLUDEDIR/$LIBRARY_NAME -Xcompiler -fopenmp your_file.cpp -L$LIBDIR -lpineapple_cuda -lcudart -fopenmp -o your_program"
        echo ""
        echo "4. For CPU-only compilation (without CUDA features):"
        echo "   g++ -std=c++20 -I$INCLUDEDIR/$LIBRARY_NAME -fopenmp your_file.cpp -fopenmp -o your_program"
    else
        echo "2. Compile using pkg-config (recommended):"
        echo "   g++ -std=c++20 \$(pkg-config --cflags pineapple) your_file.cpp \$(pkg-config --libs pineapple) -o your_program"
        echo ""
        echo "3. Or compile manually:"
        echo "   g++ -std=c++20 -I$INCLUDEDIR/$LIBRARY_NAME -fopenmp your_file.cpp -fopenmp -o your_program"
        echo ""
        echo "4. For CUDA support, reinstall with:"
        echo "   CUDA=1 ./install.sh"
    fi
    
    echo ""
    echo "5. Usage example:"
    cat << 'EOF'
   #include <pineapple/pineapple.hpp>
   
   int main() {
       Tensor<float> tensor(Shape{3, 3}, {1,2,3,4,5,6,7,8,9});
       
       #ifdef PINEAPPLE_CUDA_ENABLED
       // CUDA operations available
       tensor.to(Device::GPU);
       #endif
       
       std::cout << tensor << std::endl;
       return 0;
   }
EOF
    echo ""
    print_info "Documentation: Check headers in $INCLUDEDIR/$LIBRARY_NAME/"
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        print_info "CUDA libraries installed in: $LIBDIR/"
        print_info "To uninstall: ./uninstall.sh or manually remove files"
    else
        print_info "To uninstall: ./uninstall.sh or sudo rm -rf $INCLUDEDIR/$LIBRARY_NAME $PKGCONFIGDIR/pineapple.pc"
    fi
}

main() {
    echo ""
    print_info "=== PINEAPPLE LIBRARY INSTALLER ==="
    echo ""
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        print_info "CUDA support: ENABLED"
    else
        print_info "CUDA support: DISABLED (use CUDA=1 to enable)"
    fi
    echo ""
    
    if [ ! -f "pineapple.hpp" ]; then
        print_error "pineapple.hpp file not found in root!"
        print_error "Run this script in the Pineapple library root directory"
        exit 1
    fi
    
    if [ ! -d "inc" ]; then
        print_error "inc/ directory not found!"
        print_error "Run this script in the Pineapple library root directory"
        exit 1
    fi
    
    if [ ! -d "src" ]; then
        print_error "src/ directory not found!"
        print_error "Run this script in the Pineapple library root directory"
        exit 1
    fi
    
    # Check for CUDA source files if CUDA is enabled
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        if [ ! -f "src/tensor/tensor_cuda_kernels.cu" ] || [ ! -f "src/tensor/tensor_cuda_wrappers.cu" ]; then
            print_error "CUDA source files not found!"
            print_error "Required files: src/tensor/tensor_cuda_kernels.cu, src/tensor/tensor_cuda_wrappers.cu"
            exit 1
        fi
    fi
    
    check_permissions
    check_dependencies
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        compile_cuda_libraries
    fi
    
    install_library
    test_installation
    show_usage_info
    
    echo ""
    print_success "Installation complete!"
}

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Pineapple Library Installer"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help"
    echo ""
    echo "Environment variables:"
    echo "  PREFIX         Installation directory (default: /usr/local)"
    echo "  CUDA           Enable CUDA support (0|1, default: 0)"
    echo "  CUDA_ARCH      CUDA architectures to compile for (default: sm_50,sm_60,sm_70,sm_75,sm_80,sm_86)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Default installation to /usr/local (CPU-only)"
    echo "  PREFIX=/opt $0        # Installation to /opt (CPU-only)"
    echo "  CUDA=1 $0            # Installation with CUDA support"
    echo "  CUDA=1 PREFIX=/opt $0 # Installation to /opt with CUDA support"
    echo ""
    echo "Requirements:"
    echo "  - g++ with C++20 support"
    echo "  - OpenMP support (recommended)"
    echo "  - CUDA Toolkit (if CUDA=1)"
    echo ""
    echo "Note: CUDA support requires NVIDIA GPU and CUDA Toolkit installed"
    exit 0
fi

main
