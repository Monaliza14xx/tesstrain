# CUDA and Performance Optimization Guide

This guide explains how to optimize Tesseract LSTM training performance using GPU acceleration (CUDA) and CPU parallelization (OpenMP).

## Table of Contents

- [Overview](#overview)
- [GPU/CUDA Support](#gpucuda-support)
- [CPU Optimization](#cpu-optimization)
- [Performance Tuning](#performance-tuning)
- [Building Tesseract with CUDA](#building-tesseract-with-cuda)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

The training workflow now supports several performance optimization options:

- **GPU/CUDA acceleration**: For CUDA-enabled Tesseract builds
- **OpenMP parallelization**: Multi-threaded CPU training
- **Training parameter optimization**: Fine-tune learning rate, iterations, and checkpoints

## GPU/CUDA Support

### Prerequisites

To use GPU acceleration, you need Tesseract compiled with CUDA support. See [Building Tesseract with CUDA](#building-tesseract-with-cuda) below.

### Environment Variables

The training workflow uses the following environment variables to enable GPU acceleration:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | 0 | Enable GPU acceleration (set to 1 to enable) |
| `GPU_DEVICE` | 0 | GPU device ID for multi-GPU systems |
| `CUDA_VISIBLE_DEVICES` | Auto-set | Controls which GPUs are visible to the training process |
| `TESSERACT_OPENCL_DEVICE` | Auto-set | Tells Tesseract which OpenCL device to use (format: GPU:N) |

### GPU Optimization Parameters

New parameters for optimizing GPU memory and performance:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_MAX_MEMORY` | 12000 | Maximum GPU memory in MB. For T4 (16GB): 12000-14000 recommended |
| `NET_MODE` | 1 | LSTM network mode: 0=serial (slower, less memory), 1=parallel (faster, more memory) |
| `APPEND_INDEX` | -1 | Multi-GPU training index: -1=auto, 0+=specific index |

**Important**: When `USE_GPU=1`, the workflow automatically sets:
- `CUDA_VISIBLE_DEVICES=$(GPU_DEVICE)` - Makes only the specified GPU visible
- `TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE)` - Tells Tesseract to use the GPU
- `OMP_THREAD_LIMIT=1` - Prevents CPU multi-threading fallback
- `--max_image_MB $(GPU_MAX_MEMORY)` - Limits GPU memory usage
- `--net_mode $(NET_MODE)` - Sets LSTM network mode for performance

### Basic GPU Usage

**Important**: When `USE_GPU=1` is set, the training workflow **forces GPU-only mode** by:
- Setting `OMP_THREAD_LIMIT=1` to prevent CPU fallback
- Setting `CUDA_VISIBLE_DEVICES` to the specified GPU device
- Setting `TESSERACT_OPENCL_DEVICE=GPU:N` to tell Tesseract to use the GPU (OpenCL acceleration)
- Limiting OpenMP to 1 thread to ensure GPU is used exclusively

This ensures training runs exclusively on GPU. If the GPU is unavailable, training will fail rather than fall back to CPU. This is intentional to guarantee you're getting GPU performance.

**Note**: Tesseract uses OpenCL for GPU acceleration. The `TESSERACT_OPENCL_DEVICE` environment variable is critical - it tells Tesseract which device to use. Without it, Tesseract may default to CPU even with GPU available.

```bash
# Train with GPU acceleration (GPU-only mode, no CPU fallback)
make training MODEL_NAME=mymodel USE_GPU=1

# Use specific GPU device (for multi-GPU systems)
make training MODEL_NAME=mymodel USE_GPU=1 GPU_DEVICE=1

# Fine-tune with GPU (GPU-only mode)
make training MODEL_NAME=mymodel START_MODEL=eng USE_GPU=1

# Optimize for T4 GPU (16GB) - recommended settings
make training MODEL_NAME=mymodel USE_GPU=1 GPU_MAX_MEMORY=14000 NET_MODE=1

# Conservative T4 settings (if experiencing OOM errors)
make training MODEL_NAME=mymodel USE_GPU=1 GPU_MAX_MEMORY=12000 NET_MODE=0
```

### T4 GPU Optimization Guide

NVIDIA T4 GPUs have 16GB of VRAM. Here are recommended settings:

**Optimal Performance (T4 with 16GB)**:
```bash
make training MODEL_NAME=lao140k \
  USE_GPU=1 \
  GPU_MAX_MEMORY=14000 \
  NET_MODE=1 \
  START_MODEL=Lao \
  LEARNING_RATE=0.0001 \
  MAX_ITERATIONS=10000
```

**Memory-Constrained (if OOM errors occur)**:
```bash
make training MODEL_NAME=lao140k \
  USE_GPU=1 \
  GPU_MAX_MEMORY=12000 \
  NET_MODE=0 \
  START_MODEL=Lao \
  LEARNING_RATE=0.0001 \
  MAX_ITERATIONS=10000
```

**Parameter Explanation for T4**:
- `GPU_MAX_MEMORY=14000`: Uses 14GB of T4's 16GB VRAM, leaving 2GB for system overhead
- `NET_MODE=1`: Parallel mode - faster training, uses more memory
- `NET_MODE=0`: Serial mode - slower training, uses less memory (if experiencing OOM)

**Note**: If your Tesseract build doesn't have GPU support, training will fail rather than fall back to CPU. This is intentional to ensure you're getting the expected performance.

## CPU Optimization

### OpenMP Parallelization

Even without GPU support, you can significantly improve training performance using OpenMP multi-threading.

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | Number of CPU cores | OpenMP thread count for parallel processing |

### Usage

```bash
# Use all available CPU cores (default)
make training MODEL_NAME=mymodel

# Explicitly set thread count
make training MODEL_NAME=mymodel OMP_NUM_THREADS=8

# Limit threads to avoid resource contention
make training MODEL_NAME=mymodel OMP_NUM_THREADS=4
```

## Performance Tuning

### Training Parameters

Optimize training performance using these parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ITERATIONS` | 10000 | Number of training iterations |
| `DEBUG_INTERVAL` | 0 | Checkpoint save interval (0=only save best) |
| `LEARNING_RATE` | 0.002 (scratch) / 0.0001 (fine-tune) | Training learning rate |

```bash
# Combine optimizations for maximum performance
make training MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  OMP_NUM_THREADS=8 \
  DEBUG_INTERVAL=1000 \
  MAX_ITERATIONS=100000

# Fine-tuning with optimizations
make training MODEL_NAME=mymodel \
  START_MODEL=eng \
  USE_GPU=1 \
  LEARNING_RATE=0.0001 \
  MAX_ITERATIONS=50000
```

## Building Tesseract with GPU Support

### Important Note: OpenCL vs CUDA

**Tesseract uses OpenCL for GPU acceleration, not CUDA directly.** While CUDA is NVIDIA's proprietary framework, OpenCL is an open standard that works with NVIDIA, AMD, and Intel GPUs.

When we refer to "CUDA support" in this guide, we mean GPU acceleration through OpenCL, which can run on CUDA-compatible NVIDIA GPUs.

### Option 1: Build from Source with OpenCL Support

To enable GPU acceleration in Tesseract:

```bash
# Install OpenCL development files
# For NVIDIA GPUs:
sudo apt-get install nvidia-opencl-dev ocl-icd-opencl-dev

# For AMD GPUs:
sudo apt-get install mesa-opencl-icd ocl-icd-opencl-dev

# Clone Tesseract repository
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract

# Configure with OpenCL support
./autogen.sh
./configure --enable-opencl

# Build and install
make -j$(nproc)
sudo make install
```

### Option 2: Build with Legacy CUDA Support (Deprecated)

Note: Direct CUDA support (--enable-cuda) is deprecated. Use OpenCL instead:

```bash
# This is for reference only - OpenCL is recommended
./configure --enable-cuda
```

### Option 2: Using Pre-built CUDA-enabled Tesseract

Some distributions provide CUDA-enabled builds. Check your package manager or Tesseract releases.

### Verify CUDA Support

After building, verify CUDA/OpenCL support:

```bash
# Check if OpenCL is available in Tesseract
lstmtraining --help 2>&1 | grep -i opencl

# Check available OpenCL devices
clinfo

# Test GPU is accessible
nvidia-smi

# Verify GPU is being used during training
# In one terminal, start training:
make training MODEL_NAME=test USE_GPU=1

# In another terminal, monitor GPU usage:
watch -n 1 nvidia-smi
# You should see GPU utilization increase and memory usage grow
```

### Troubleshooting GPU Detection

If training still uses CPU despite `USE_GPU=1`:

1. **Check OpenCL Support in Tesseract**:
   ```bash
   # lstmtraining should mention OpenCL if compiled with support
   lstmtraining --help 2>&1 | head -20
   ```

2. **List Available OpenCL Devices**:
   ```bash
   # Install clinfo if not available
   sudo apt-get install clinfo
   clinfo
   ```

3. **Check Environment Variables Are Set**:
   Look for these in the training output:
   - `CUDA_VISIBLE_DEVICES=0`
   - `TESSERACT_OPENCL_DEVICE=GPU:0`
   - `OMP_THREAD_LIMIT=1`

4. **Monitor GPU Usage**:
   ```bash
   # Run this while training is active
   nvidia-smi
   # Look for:
   # - GPU utilization > 0%
   # - Memory usage increasing
   # - Process 'lstmtraining' listed
   ```

## Usage Examples

### Example 1: Training from Scratch with GPU

```bash
# Prepare ground truth data
unzip your-data.zip -d data/mymodel-ground-truth

# Train with GPU acceleration
make training MODEL_NAME=mymodel \
  USE_GPU=1 \
  OMP_NUM_THREADS=8 \
  MAX_ITERATIONS=100000 \
  LEARNING_RATE=0.002
```

### Example 2: Fine-tuning with Optimizations

```bash
# Fine-tune existing model with GPU
make training MODEL_NAME=mymodel \
  START_MODEL=eng \
  TESSDATA=~/tessdata_best \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  LEARNING_RATE=0.0001 \
  MAX_ITERATIONS=50000 \
  TARGET_ERROR_RATE=0.005
```

### Example 3: CPU-only Optimization

```bash
# Optimize for CPU-only systems
make training MODEL_NAME=mymodel \
  OMP_NUM_THREADS=$(nproc) \
  MAX_ITERATIONS=50000 \
  DEBUG_INTERVAL=1000
```

### Example 4: Multi-GPU System

```bash
# Train multiple models on different GPUs simultaneously
make training MODEL_NAME=model1 USE_GPU=1 GPU_DEVICE=0 &
make training MODEL_NAME=model2 USE_GPU=1 GPU_DEVICE=1 &
wait
```

## Troubleshooting

### GPU Not Being Used (Training Falls Back to CPU)

**Problem**: Training configuration shows "GPU ENABLED" but training still uses CPU

**Root Cause**: The `TESSERACT_OPENCL_DEVICE` environment variable was missing. Without this variable, Tesseract doesn't know which GPU device to use, even if CUDA_VISIBLE_DEVICES is set.

**Solution**: As of the latest update, `USE_GPU=1` now **automatically sets all required environment variables**:
- `OMP_THREAD_LIMIT=1` - Prevents CPU multi-threading fallback
- `CUDA_VISIBLE_DEVICES=$(GPU_DEVICE)` - Makes GPU visible to CUDA runtime
- `TESSERACT_OPENCL_DEVICE=GPU:$(GPU_DEVICE)` - **Critical**: Tells Tesseract which GPU to use

**Verify GPU is being used**:
1. Check training output includes all three environment variables
2. Run `nvidia-smi` in another terminal during training
3. Look for GPU utilization > 0% and increasing memory usage
4. Verify `lstmtraining` process is listed in GPU processes

**Example corrected command**:
```bash
OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=0 TESSERACT_OPENCL_DEVICE=GPU:0 \
lstmtraining \
  --traineddata model.traineddata \
  --train_listfile list.train \
  --eval_listfile list.eval \
  --max_iterations 10000
```

### GPU Not Detected

**Problem**: Training fails immediately with GPU errors

**Solutions**:
1. Verify Tesseract is built with CUDA: `lstmtraining --help | grep -i cuda`
2. Check CUDA installation: `nvidia-smi`
3. Verify CUDA_VISIBLE_DEVICES is set correctly
4. Check GPU memory availability
5. If Tesseract doesn't have GPU support, use `USE_GPU=0` for CPU training

### Out of Memory Errors

**Problem**: Training crashes with CUDA out of memory

**Solutions**:
1. Reduce image resolution in ground truth data
2. Close other GPU applications
3. Monitor GPU memory: `nvidia-smi`
4. Use a GPU with more VRAM

### Slow Training Performance

**Problem**: Training is slower than expected

**Solutions**:
1. Verify GPU is actually being used: `nvidia-smi` during training
2. Enable GPU if available: `USE_GPU=1`
3. Check DEBUG_INTERVAL isn't too frequent (use 1000-5000)
4. Ensure data is on fast storage (SSD preferred)
5. For CPU: Verify OMP_NUM_THREADS is set appropriately

### OpenMP Errors

**Problem**: OpenMP-related warnings or errors

**Solutions**:
1. Install OpenMP: `sudo apt-get install libomp-dev`
2. Rebuild Tesseract with OpenMP support
3. Set OMP_NUM_THREADS explicitly: `OMP_NUM_THREADS=4`

## Performance Benchmarks

Typical speedup factors (compared to single-threaded CPU):

| Configuration | Relative Speed | Notes |
|--------------|----------------|-------|
| Single CPU thread | 1x | Baseline |
| 8 CPU threads (OpenMP) | 3-5x | Depends on CPU architecture |
| GPU (CUDA) | 10-20x | Depends on GPU model |
| GPU + optimizations | 15-30x | GPU + large batch size |

**Note**: Actual performance depends on:
- Hardware specifications (CPU, GPU model)
- Dataset characteristics (image size, complexity)
- Network architecture (NET_SPEC)
- Batch size and other parameters

## Best Practices

1. **Start with defaults**: Use default settings first to establish a baseline
2. **Incremental optimization**: Change one parameter at a time
3. **Monitor GPU usage**: Use `nvidia-smi` to verify GPU utilization
4. **Balance memory/speed**: Increase batch size until memory is nearly full
5. **Use checkpoints**: Set DEBUG_INTERVAL to save progress regularly
6. **Profile performance**: Compare training times with different settings

## Additional Resources

- [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- [LSTM Training Guide](https://tesseract-ocr.github.io/tessdoc/tess5/TrainingTesseract-5.html)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- [OpenMP Documentation](https://www.openmp.org/)

## Contributing

Found a way to improve performance? Please contribute your findings by opening an issue or pull request!
