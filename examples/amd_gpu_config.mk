# Example Configuration for AMD GPU-Accelerated Training
# AMD GPU with ROCm/OpenCL

# This configuration demonstrates how to use GPU acceleration with AMD GPUs
# Prerequisites:
# - AMD GPU (RX series, Radeon VII, MI series, etc.)
# - ROCm or AMD APP SDK installed
# - Tesseract compiled with OpenCL support

# Model configuration
MODEL_NAME=my_amd_model
DATA_DIR=data
GROUND_TRUTH_DIR=data/my_amd_model-ground-truth

# Training parameters
MAX_ITERATIONS=10000
LEARNING_RATE=0.0001
RATIO_TRAIN=0.90

# GPU Acceleration with AMD
# Enable OpenCL for GPU acceleration
USE_OPENCL=yes

# CPU thread count - use a moderate number
OPENMP_THREAD_COUNT=4

# Usage:
# make training -f examples/amd_gpu_config.mk

# To check if OpenCL is working:
# 1. Run: ./check_acceleration.sh
# 2. Verify AMD device is listed
# 3. Use clinfo to see device details

# Performance tips:
# - Monitor GPU with: radeontop or rocm-smi
# - Ensure ROCm version is compatible with your GPU
# - Some older AMD GPUs may need legacy APP SDK
# - Check AMD documentation for OpenCL optimization
