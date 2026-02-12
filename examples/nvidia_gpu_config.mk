# Example Configuration for GPU-Accelerated Training
# NVIDIA GPU with CUDA/OpenCL

# This configuration demonstrates how to use GPU acceleration with NVIDIA GPUs
# Prerequisites:
# - NVIDIA GPU (GTX, RTX, Tesla, etc.)
# - CUDA Toolkit installed (includes OpenCL)
# - Tesseract compiled with OpenCL support

# Model configuration
MODEL_NAME=my_nvidia_model
DATA_DIR=data
GROUND_TRUTH_DIR=data/my_nvidia_model-ground-truth

# Training parameters
MAX_ITERATIONS=10000
LEARNING_RATE=0.0001
RATIO_TRAIN=0.90

# GPU Acceleration with NVIDIA
# Enable OpenCL for GPU acceleration
USE_OPENCL=yes

# CPU thread count - use a moderate number to avoid CPU bottleneck
# Even with GPU, some CPU threads help with data loading
OPENMP_THREAD_COUNT=4

# Usage:
# make training -f examples/nvidia_gpu_config.mk

# To check if OpenCL is working:
# 1. Run: ./check_acceleration.sh
# 2. Verify NVIDIA device is listed
# 3. Check training log for GPU usage indicators

# Performance tips:
# - Monitor GPU usage with: nvidia-smi -l 1
# - Ensure CUDA Toolkit version matches driver
# - Larger batch sizes may benefit from GPU acceleration more
# - For multiple GPUs, OpenCL may use the default device
