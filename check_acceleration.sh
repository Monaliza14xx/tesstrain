#!/usr/bin/env bash
# GPU/CPU Acceleration Detection Script for Tesstrain
# This script checks for available acceleration options (OpenCL, OpenMP)

set -e

echo "==================================================================="
echo "Tesseract Training Acceleration Detection"
echo "==================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if lstmtraining is available
echo "1. Checking for lstmtraining..."
if command -v lstmtraining &> /dev/null; then
    echo -e "${GREEN}✓ lstmtraining found${NC}"
    LSTMTRAINING_PATH=$(which lstmtraining)
    echo "  Path: $LSTMTRAINING_PATH"
else
    echo -e "${RED}✗ lstmtraining not found in PATH${NC}"
    echo "  Please install Tesseract with training tools"
    exit 1
fi

echo ""

# Check OpenMP support
echo "2. Checking for OpenMP support..."
if lstmtraining --help 2>&1 | grep -q "openmp_thread_count"; then
    echo -e "${GREEN}✓ OpenMP support available${NC}"
    echo "  lstmtraining supports --openmp_thread_count parameter"
    
    # Get number of CPU cores
    if [[ "$OSTYPE" == "darwin"* ]]; then
        NUM_CORES=$(sysctl -n hw.ncpu)
    else
        NUM_CORES=$(nproc)
    fi
    echo "  Available CPU cores: $NUM_CORES"
    echo -e "  ${YELLOW}Recommendation: Set OPENMP_THREAD_COUNT=$NUM_CORES for maximum CPU performance${NC}"
else
    echo -e "${YELLOW}! OpenMP parameter not found in lstmtraining help${NC}"
    echo "  This is normal for some Tesseract builds"
fi

echo ""

# Check for OpenCL
echo "3. Checking for OpenCL support..."

# Check if Tesseract was built with OpenCL
if ldd "$LSTMTRAINING_PATH" 2>/dev/null | grep -q "OpenCL"; then
    echo -e "${GREEN}✓ lstmtraining is linked with OpenCL${NC}"
    OPENCL_LINKED=1
elif otool -L "$LSTMTRAINING_PATH" 2>/dev/null | grep -q "OpenCL"; then
    echo -e "${GREEN}✓ lstmtraining is linked with OpenCL (macOS)${NC}"
    OPENCL_LINKED=1
else
    echo -e "${YELLOW}! lstmtraining does not appear to be linked with OpenCL${NC}"
    echo "  OpenCL acceleration will not be available"
    echo "  To enable: Rebuild Tesseract with OpenCL support"
    OPENCL_LINKED=0
fi

echo ""

# Check for OpenCL runtime and devices
echo "4. Checking for OpenCL devices..."
if command -v clinfo &> /dev/null; then
    echo -e "${GREEN}✓ clinfo tool found${NC}"
    
    # Get OpenCL device info
    DEVICE_COUNT=$(clinfo 2>/dev/null | grep -c "Device Name" || echo "0")
    
    if [ "$DEVICE_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found $DEVICE_COUNT OpenCL device(s)${NC}"
        echo ""
        echo "  Available OpenCL devices:"
        clinfo 2>/dev/null | grep -A 3 "Device Name" | head -20
    else
        echo -e "${YELLOW}! No OpenCL devices found${NC}"
        echo "  Install OpenCL runtime for your GPU:"
        echo "  - NVIDIA: Install CUDA Toolkit"
        echo "  - AMD: Install ROCm or AMD APP SDK"
        echo "  - Intel: Install Intel OpenCL Runtime"
    fi
else
    echo -e "${YELLOW}! clinfo not found${NC}"
    echo "  Install clinfo to check for OpenCL devices:"
    echo "  - Ubuntu/Debian: sudo apt-get install clinfo"
    echo "  - Fedora/RHEL: sudo dnf install clinfo"
    echo "  - macOS: OpenCL is built-in, but clinfo may not be available"
fi

echo ""
echo "==================================================================="
echo "Summary and Recommendations"
echo "==================================================================="
echo ""

# Provide recommendations based on findings
if [ "$OPENCL_LINKED" -eq 1 ] && [ "$DEVICE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}GPU Acceleration: AVAILABLE${NC}"
    echo "  You can enable GPU acceleration by setting:"
    echo "  USE_OPENCL=yes make training MODEL_NAME=yourmodel"
    echo ""
elif [ "$OPENCL_LINKED" -eq 1 ]; then
    echo -e "${YELLOW}GPU Acceleration: PARTIALLY AVAILABLE${NC}"
    echo "  lstmtraining has OpenCL support, but no devices detected"
    echo "  Install OpenCL runtime for your GPU"
    echo ""
elif [ "$DEVICE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}GPU Acceleration: NOT AVAILABLE${NC}"
    echo "  OpenCL devices found, but lstmtraining not built with OpenCL"
    echo "  Rebuild Tesseract with OpenCL support"
    echo ""
else
    echo -e "${YELLOW}GPU Acceleration: NOT AVAILABLE${NC}"
    echo "  Neither OpenCL support nor devices detected"
    echo ""
fi

# CPU parallelization is always available
echo -e "${GREEN}CPU Parallelization: AVAILABLE${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    NUM_CORES=$(nproc 2>/dev/null || echo "N/A")
fi
echo "  Set OPENMP_THREAD_COUNT=$NUM_CORES for optimal CPU performance"
echo ""

echo "Example usage:"
echo "  # CPU-only with 8 threads:"
echo "  make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=8"
echo ""
echo "  # GPU acceleration (if available):"
echo "  make training MODEL_NAME=mymodel USE_OPENCL=yes"
echo ""
echo "  # Both CPU and GPU:"
echo "  make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=8 USE_OPENCL=yes"
echo ""
