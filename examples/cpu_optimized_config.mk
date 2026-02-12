# Example Configuration for CPU-Only Optimized Training
# Maximum CPU Performance

# This configuration demonstrates optimal CPU-only training
# No GPU required - uses OpenMP for multi-threaded CPU parallelization

# Model configuration
MODEL_NAME=my_cpu_model
DATA_DIR=data
GROUND_TRUTH_DIR=data/my_cpu_model-ground-truth

# Training parameters
MAX_ITERATIONS=10000
LEARNING_RATE=0.0001
RATIO_TRAIN=0.90

# CPU Optimization
# Use all available CPU cores for training
# Replace with specific number if needed (e.g., 8, 16, 32)
OPENMP_THREAD_COUNT=$(shell nproc)

# GPU acceleration disabled (default)
USE_OPENCL=

# Usage:
# make training -f examples/cpu_optimized_config.mk

# Alternative usage with explicit thread count:
# make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=16

# Performance tips:
# - Monitor CPU usage with: htop or top
# - Ensure system has adequate RAM (2-4GB per thread recommended)
# - For large datasets, consider using swap space
# - Close other CPU-intensive applications during training
# - Consider using taskset to bind to specific CPU cores for consistency

# Advanced: Parallel data preparation AND training
# make -j$(nproc) training OPENMP_THREAD_COUNT=$(nproc)
# This uses all cores for both file generation and training
