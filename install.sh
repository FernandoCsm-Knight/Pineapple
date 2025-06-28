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
PKGCONFIGDIR="$PREFIX/lib/pkgconfig"

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
    
    # Check OpenMP
    if ! echo '#include <omp.h>' | g++ -fopenmp -x c++ - -o /tmp/test_omp 2>/dev/null; then
        print_warning "OpenMP may not be available. Library will work but without parallelization"
    else
        rm -f /tmp/test_omp
    fi
    
    print_success "Dependencies checked"
}

install_library() {
    print_info "Installing Pineapple library..."
    
    print_info "Creating installation directories..."
    $SUDO mkdir -p "$INCLUDEDIR/$LIBRARY_NAME"
    $SUDO mkdir -p "$PKGCONFIGDIR"
    
    print_info "Copying header files and implementations..."
    $SUDO cp -r inc "$INCLUDEDIR/$LIBRARY_NAME/"
    $SUDO cp -r src "$INCLUDEDIR/$LIBRARY_NAME/"
    $SUDO cp pineapple.hpp "$INCLUDEDIR/$LIBRARY_NAME/"
    
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
    
    $SUDO cp /tmp/pineapple.pc "$PKGCONFIGDIR/"
    rm -f /tmp/pineapple.pc
    
    # Set appropriate permissions
    print_info "Setting permissions..."
    $SUDO chmod -R 644 "$INCLUDEDIR/$LIBRARY_NAME"
    $SUDO find "$INCLUDEDIR/$LIBRARY_NAME" -type d -exec chmod 755 {} \;
    $SUDO chmod 644 "$PKGCONFIGDIR/pineapple.pc"
    
    print_success "Pineapple library installed successfully!"
}

test_installation() {
    print_info "Testing installation..."
    
    # Create test file
    cat > /tmp/test_pineapple.cpp << 'EOF'
#include <pineapple/pineapple.hpp>
#include <iostream>

int main() {
    Tensor<float> tensor(Shape{2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Pineapple library test: OK" << std::endl;
    std::cout << "Tensor created successfully: " << tensor.length() << " elements" << std::endl;
    return 0;
}
EOF
    
    # Try to compile
    if pkg-config --exists pineapple 2>/dev/null; then
        if g++ -std=c++20 $(pkg-config --cflags pineapple) /tmp/test_pineapple.cpp $(pkg-config --libs pineapple) -o /tmp/test_pineapple 2>/dev/null; then
            if /tmp/test_pineapple >/dev/null 2>&1; then
                print_success "Installation test passed!"
            else
                print_warning "Compilation OK, but execution failed"
            fi
        else
            print_warning "Test compilation failed"
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
    echo "2. Compile using pkg-config (recommended):"
    echo "   g++ -std=c++20 \$(pkg-config --cflags pineapple) your_file.cpp \$(pkg-config --libs pineapple) -o your_program"
    echo ""
    echo "3. Or compile manually:"
    echo "   g++ -std=c++20 -I$INCLUDEDIR/$LIBRARY_NAME -fopenmp your_file.cpp -fopenmp -o your_program"
    echo ""
    echo "4. Usage example:"
    cat << 'EOF'
   #include <pineapple/pineapple.hpp>
   
   int main() {
       Tensor<float> tensor(Shape{3, 3}, {1,2,3,4,5,6,7,8,9});
       std::cout << tensor << std::endl;
       return 0;
   }
EOF
    echo ""
    print_info "Documentation: Check headers in $INCLUDEDIR/$LIBRARY_NAME/"
    print_info "To uninstall: ./uninstall.sh or sudo rm -rf $INCLUDEDIR/$LIBRARY_NAME $PKGCONFIGDIR/pineapple.pc"
}

main() {
    echo ""
    print_info "=== PINEAPPLE LIBRARY INSTALLER ==="
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
    
    check_permissions
    check_dependencies
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
    echo ""
    echo "Examples:"
    echo "  $0                    # Default installation to /usr/local"
    echo "  PREFIX=/opt $0        # Installation to /opt"
    exit 0
fi

main
