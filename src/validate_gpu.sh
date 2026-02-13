#!/bin/bash
# GPU Performance Validation Script
# This script helps validate that GPU acceleration is working properly

echo "=== GPU Performance Validation ==="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU monitoring not available."
    echo "   Install NVIDIA drivers to enable GPU monitoring."
    exit 1
fi

echo "✓ nvidia-smi found"
echo ""

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Check for OpenCL support
echo "=== Checking OpenCL Support ==="
if command -v clinfo &> /dev/null; then
    echo "✓ clinfo found"
    OPENCL_DEVICES=$(clinfo 2>/dev/null | grep -c "Device Type.*GPU")
    echo "   OpenCL GPU devices found: $OPENCL_DEVICES"
else
    echo "⚠ clinfo not found. Install with: sudo apt-get install clinfo"
    echo "   This is needed to verify OpenCL GPU support."
fi
echo ""

# Check if lstmtraining is available
echo "=== Checking lstmtraining ==="
if command -v lstmtraining &> /dev/null; then
    echo "✓ lstmtraining found"
    # Try to detect if it has OpenCL support
    if lstmtraining --help 2>&1 | grep -q "OpenCL\|opencl"; then
        echo "✓ lstmtraining appears to have OpenCL support"
    else
        echo "⚠ Could not detect OpenCL support in lstmtraining"
        echo "   lstmtraining may need to be rebuilt with --enable-opencl"
    fi
else
    echo "❌ lstmtraining not found in PATH"
    echo "   Make sure Tesseract is installed"
fi
echo ""

# Provide recommendations
echo "=== Recommendations ==="
echo ""
echo "To use GPU acceleration with tesstrain:"
echo "  1. Set USE_GPU=1 in your make command"
echo "  2. Set GPU_MAX_MEMORY based on your GPU memory:"
for gpu_mem in $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null); do
    recommended=$((gpu_mem * 80 / 100))
    echo "     GPU Memory: ${gpu_mem}MB → Recommended GPU_MAX_MEMORY=${recommended}"
done
echo "  3. Use NET_MODE=1 for parallel processing (faster, more memory)"
echo "  4. Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "Example command:"
echo "  make training MODEL_NAME=mymodel USE_GPU=1 GPU_MAX_MEMORY=14000 NET_MODE=1"
echo ""

# Test GPU availability during training
echo "=== GPU Availability Test ==="
echo "Current GPU utilization:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader
echo ""
echo "If GPU utilization is 0% during training, check:"
echo "  1. Tesseract built with --enable-opencl"
echo "  2. TESSERACT_OPENCL_DEVICE environment variable is set"
echo "  3. No other processes are using the GPU"
echo ""

echo "=== Validation Complete ==="
