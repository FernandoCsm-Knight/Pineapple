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

# CUD    if [[ "$CUDA_ENABLED" == "1" ]]; then
        echo "2. Compile with CUDA support using pkg-config (recommended):"
        echo "   nvcc --extended-lambda \$(pkg-config --cflags pineapple) -Xcompiler -fopenmp your_file.cpp \$(pkg-config --libs pineapple) -o your_program"
        echo ""
        echo "3. Or compile manually with CUDA:"
        echo "   nvcc --extended-lambda -std=c++20 -I$INCLUDEDIR -Xcompiler -fopenmp your_file.cpp -L$LIBDIR -lpineapple -lcudart -o your_program"
        echo ""
        echo "4. For CPU-only compilation (without CUDA features):"
        echo "   g++ -std=c++20 -I$INCLUDEDIR -fopenmp your_file.cpp -fopenmp -o your_program"ration
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

compile_libraries() {
    print_info "Compiling Pineapple libraries..."
    
    # Create temporary build directory
    BUILD_DIR="/tmp/pineapple_build"
    mkdir -p "$BUILD_DIR"
    
    # Copy source files to build directory
    cp -r src "$BUILD_DIR/"
    cp -r inc "$BUILD_DIR/"
    
    cd "$BUILD_DIR"
    
    # Find all CUDA and C++ source files
    CUDA_FILES=$(find src -name "*.cu" | sort)
    CPP_FILES=$(find src -name "*.cpp" | sort)
    
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        if [[ -z "$CUDA_FILES" && -z "$CPP_FILES" ]]; then
            print_error "No CUDA (.cu) or C++ (.cpp) source files found in src/"
            exit 1
        fi
        
        # CUDA compilation flags
        COMPILER_FLAGS="--extended-lambda -std=c++20 \
                        -Wno-deprecated-gpu-targets \
                        -gencode arch=compute_50,code=sm_50 \
                        -gencode arch=compute_60,code=sm_60 \
                        -gencode arch=compute_70,code=sm_70 \
                        -gencode arch=compute_75,code=sm_75 \
                        -gencode arch=compute_80,code=sm_80 \
                        -gencode arch=compute_86,code=sm_86 \
                        -Xcompiler -fPIC,-fopenmp \
                        -Iinc"
        
        # Compile CUDA files
        OBJECT_FILES=""
        if [[ -n "$CUDA_FILES" ]]; then
            print_info "Compiling CUDA source files..."
            for cu_file in $CUDA_FILES; do
                obj_name=$(basename "${cu_file%.cu}.o")
                print_info "  Compiling $cu_file -> $obj_name"
                nvcc $COMPILER_FLAGS -c "$cu_file" -o "$obj_name"
                OBJECT_FILES="$OBJECT_FILES $obj_name"
            done
        fi
        
        # Compile C++ files with nvcc for consistency
        if [[ -n "$CPP_FILES" ]]; then
            print_info "Compiling C++ source files..."
            for cpp_file in $CPP_FILES; do
                obj_name=$(basename "${cpp_file%.cpp}.o")
                print_info "  Compiling $cpp_file -> $obj_name"
                nvcc $COMPILER_FLAGS -c "$cpp_file" -o "$obj_name"
                OBJECT_FILES="$OBJECT_FILES $obj_name"
            done
        fi
        
        if [[ -z "$OBJECT_FILES" ]]; then
            print_error "No object files were generated"
            exit 1
        fi
        
        # Create static library
        print_info "Creating static library..."
        ar rcs libpineapple.a $OBJECT_FILES
        
        # Create shared library
        print_info "Creating shared library..."
        nvcc -shared -Xcompiler -fPIC \
             $OBJECT_FILES \
             -o libpineapple.so.${VERSION} \
             -lcudart
        
        # Create version symlinks
        ln -sf libpineapple.so.${VERSION} libpineapple.so.${VERSION%.*}
        ln -sf libpineapple.so.${VERSION} libpineapple.so
        
        print_success "CUDA libraries compiled successfully"
        print_info "Compiled files: $(echo $CUDA_FILES $CPP_FILES | wc -w) source files -> $(echo $OBJECT_FILES | wc -w) object files"
    else
        # CPU-only compilation
        if [[ -z "$CPP_FILES" ]]; then
            print_warning "No C++ (.cpp) source files found in src/ for CPU-only build"
            print_info "CPU-only build will be header-only"
            cd - > /dev/null
            return 0
        fi
        
        # GCC compilation flags for CPU-only
        COMPILER_FLAGS="-std=c++20 -fPIC -fopenmp -O3 -Iinc"
        
        # Compile C++ files with g++
        OBJECT_FILES=""
        print_info "Compiling C++ source files for CPU-only build..."
        for cpp_file in $CPP_FILES; do
            obj_name=$(basename "${cpp_file%.cpp}.o")
            print_info "  Compiling $cpp_file -> $obj_name"
            g++ $COMPILER_FLAGS -c "$cpp_file" -o "$obj_name"
            OBJECT_FILES="$OBJECT_FILES $obj_name"
        done
        
        if [[ -z "$OBJECT_FILES" ]]; then
            print_warning "No object files were generated for CPU-only build"
            cd - > /dev/null
            return 0
        fi
        
        # Create static library
        print_info "Creating static library..."
        ar rcs libpineapple.a $OBJECT_FILES
        
        # Create shared library
        print_info "Creating shared library..."
        g++ -shared -fPIC $OBJECT_FILES -o libpineapple.so.${VERSION} -fopenmp
        
        # Create version symlinks
        ln -sf libpineapple.so.${VERSION} libpineapple.so.${VERSION%.*}
        ln -sf libpineapple.so.${VERSION} libpineapple.so
        
        print_success "CPU libraries compiled successfully"
        print_info "Compiled files: $(echo $CPP_FILES | wc -w) source files -> $(echo $OBJECT_FILES | wc -w) object files"
    fi
    
    # Move back to original directory
    cd - > /dev/null
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
    
    # Install compiled libraries if they exist
    if [[ -f "/tmp/pineapple_build/libpineapple.a" ]]; then
        if [[ "$CUDA_ENABLED" == "1" ]]; then
            print_info "Installing CUDA libraries..."
        else
            print_info "Installing CPU libraries..."
        fi
        
        $SUDO cp /tmp/pineapple_build/libpineapple.a "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple.so.${VERSION} "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple.so.${VERSION%.*} "$LIBDIR/"
        $SUDO cp /tmp/pineapple_build/libpineapple.so "$LIBDIR/"
        
        # Update library cache
        if command -v ldconfig >/dev/null 2>&1; then
            $SUDO ldconfig
        fi
        
        if [[ "$CUDA_ENABLED" == "1" ]]; then
            print_success "CUDA libraries installed"
        else
            print_success "CPU libraries installed"
        fi
    else
        if [[ "$CUDA_ENABLED" == "1" ]]; then
            print_warning "No CUDA libraries were compiled, installation will be header-only"
        else
            print_info "No compiled libraries needed, installation is header-only"
        fi
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
Cflags: -I\${includedir} -std=c++20
Libs: -L\${libdir} -lpineapple -lcudart
Libs.private: -lcuda
EOF
    else
        # pkg-config without CUDA support
        if [[ -f "/tmp/pineapple_build/libpineapple.a" ]]; then
            # CPU version with compiled library
            cat > /tmp/pineapple.pc << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include
libdir=\${prefix}/lib

Name: $LIBRARY_NAME
Description: Pineapple Neural Network Library - Header-only C++ template library
Version: $VERSION
Cflags: -I\${includedir} -fopenmp -std=c++20
Libs: -L\${libdir} -lpineapple -fopenmp
EOF
        else
            # CPU version header-only
            cat > /tmp/pineapple.pc << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
includedir=\${prefix}/include

Name: $LIBRARY_NAME
Description: Pineapple Neural Network Library - Header-only C++ template library
Version: $VERSION
Cflags: -I\${includedir} -fopenmp -std=c++20
Libs: -fopenmp
EOF
        fi
    fi
    
    $SUDO cp /tmp/pineapple.pc "$PKGCONFIGDIR/"
    rm -f /tmp/pineapple.pc
    
    # Set appropriate permissions
    print_info "Setting permissions..."
    $SUDO chmod -R 644 "$INCLUDEDIR/$LIBRARY_NAME"
    $SUDO find "$INCLUDEDIR/$LIBRARY_NAME" -type d -exec chmod 755 {} \;
    $SUDO chmod 644 "$PKGCONFIGDIR/pineapple.pc"
    
    # Set permissions for compiled libraries if they exist
    if [[ -f "$LIBDIR/libpineapple.a" ]]; then
        $SUDO chmod 644 "$LIBDIR"/libpineapple.*
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
            # Test with nvcc for CUDA (need to handle OpenMP flag specially)
            NVCC_CFLAGS=$(pkg-config --cflags pineapple)
            NVCC_LIBS=$(pkg-config --libs pineapple)
            
            if nvcc --extended-lambda $NVCC_CFLAGS -Xcompiler -fopenmp /tmp/test_pineapple.cpp $NVCC_LIBS -o /tmp/test_pineapple 2>/dev/null; then
                if /tmp/test_pineapple >/dev/null 2>&1; then
                    print_success "Installation test passed (with CUDA)!"
                else
                    print_warning "Compilation OK, but execution failed"
                fi
            else
                print_warning "Test compilation with nvcc failed - trying detailed error output..."
                # Try again with error output for debugging
                nvcc --extended-lambda $NVCC_CFLAGS -Xcompiler -fopenmp /tmp/test_pineapple.cpp $NVCC_LIBS -o /tmp/test_pineapple_debug 2>&1 | head -10
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
        echo "   nvcc --extended-lambda -std=c++20 -I$INCLUDEDIR -Xcompiler -fopenmp your_file.cpp -L$LIBDIR -lpineapple -lcudart -o your_program"
        echo ""
        echo "4. For CPU-only compilation (without CUDA features):"
        echo "   g++ -std=c++20 -I$INCLUDEDIR -fopenmp your_file.cpp -fopenmp -o your_program"
    else
        echo "2. Compile using pkg-config (recommended):"
        echo "   g++ -std=c++20 \$(pkg-config --cflags pineapple) your_file.cpp \$(pkg-config --libs pineapple) -o your_program"
        echo ""
        if [[ -f "$LIBDIR/libpineapple.a" ]]; then
            echo "3. Or compile manually:"
            echo "   g++ -std=c++20 -I$INCLUDEDIR -fopenmp your_file.cpp -L$LIBDIR -lpineapple -fopenmp -o your_program"
        else
            echo "3. Or compile manually (header-only):"
            echo "   g++ -std=c++20 -I$INCLUDEDIR -fopenmp your_file.cpp -fopenmp -o your_program"
        fi
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
    
    # Check for source files
    if [[ "$CUDA_ENABLED" == "1" ]]; then
        CUDA_FILES=$(find src -name "*.cu" 2>/dev/null)
        CPP_FILES=$(find src -name "*.cpp" 2>/dev/null)
        
        if [[ -z "$CUDA_FILES" && -z "$CPP_FILES" ]]; then
            print_error "No CUDA (.cu) or C++ (.cpp) source files found!"
            print_error "CUDA support requires at least one .cu or .cpp file in src/ directory"
            print_info "Found directories: $(find src -type d | tr '\n' ' ')"
            exit 1
        fi
        
        # Check for essential CUDA headers
        if [ ! -f "inc/device/tensor_cuda_wrappers.hpp" ]; then
            print_warning "CUDA wrapper header not found: inc/device/tensor_cuda_wrappers.hpp"
        fi
        
        print_info "Found CUDA/C++ source files:"
        if [[ -n "$CUDA_FILES" ]]; then
            print_info "  CUDA files (.cu): $(echo $CUDA_FILES | wc -w)"
            for file in $CUDA_FILES; do
                print_info "    $file"
            done
        fi
        if [[ -n "$CPP_FILES" ]]; then
            print_info "  C++ files (.cpp): $(echo $CPP_FILES | wc -w)"
            for file in $CPP_FILES; do
                print_info "    $file"
            done
        fi
    else
        # For CPU-only build, check for C++ files
        CPP_FILES=$(find src -name "*.cpp" 2>/dev/null)
        
        if [[ -n "$CPP_FILES" ]]; then
            print_info "Found C++ source files for compilation:"
            print_info "  C++ files (.cpp): $(echo $CPP_FILES | wc -w)"
            for file in $CPP_FILES; do
                print_info "    $file"
            done
        else
            print_info "No C++ source files found - installation will be header-only"
        fi
    fi
    
    check_permissions
    check_dependencies
    
    # Compile libraries for both CUDA and CPU versions
    compile_libraries
    
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
