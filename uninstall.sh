#!/bin/bash

# Pineapple library uninstallation script

set -e

# Configuration
LIBRARY_NAME="pineapple"
PREFIX="${PREFIX:-/usr/local}"
INCLUDEDIR="$PREFIX/include"
LIBDIR="$PREFIX/lib"
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
        print_info "sudo will be required for uninstallation"
    else
        print_error "You need root permissions or sudo installed"
        exit 1
    fi
}

uninstall_library() {
    print_info "Uninstalling Pineapple library..."
    
    # Check if library is installed
    if [ ! -d "$INCLUDEDIR/$LIBRARY_NAME" ] && [ ! -f "$PKGCONFIGDIR/pineapple.pc" ] && [ ! -f "$LIBDIR/libpineapple.a" ]; then
        print_warning "Pineapple library does not appear to be installed"
        return 0
    fi
    
    # Remove headers
    if [ -d "$INCLUDEDIR/$LIBRARY_NAME" ]; then
        print_info "Removing headers..."
        $SUDO rm -rf "$INCLUDEDIR/$LIBRARY_NAME"
        print_success "Headers removed"
    fi
    
    # Remove compiled libraries
    if [ -f "$LIBDIR/libpineapple.a" ] || [ -f "$LIBDIR/libpineapple.so" ]; then
        print_info "Removing compiled libraries..."
        $SUDO rm -f "$LIBDIR"/libpineapple.*
        
        # Update library cache
        if command -v ldconfig >/dev/null 2>&1; then
            $SUDO ldconfig
        fi
        
        print_success "Compiled libraries removed"
    fi
    
    # Remove pkg-config
    if [ -f "$PKGCONFIGDIR/pineapple.pc" ]; then
        print_info "Removing pkg-config configuration..."
        $SUDO rm -f "$PKGCONFIGDIR/pineapple.pc"
        print_success "pkg-config file removed"
    fi
    
    print_success "Pineapple library uninstalled successfully!"
}

main() {
    echo ""
    print_info "=== PINEAPPLE LIBRARY UNINSTALLER ==="
    echo ""
    
    check_permissions
    
    read -p "Are you sure you want to uninstall the Pineapple library? (y/N): " confirm
    
    if [[ $confirm =~ ^[Yy]$ ]]; then
        uninstall_library
        echo ""
        print_success "Uninstallation complete!"
    else
        print_info "Uninstallation cancelled"
    fi
}

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Pineapple Library Uninstaller"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help"
    echo "  --force        Remove without confirmation"
    echo ""
    echo "Environment variables:"
    echo "  PREFIX         Installation directory (default: /usr/local)"
    exit 0
fi

if [[ "$1" == "--force" ]]; then
    check_permissions
    uninstall_library
else
    main
fi
