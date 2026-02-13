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
- **Batch size tuning**: Memory vs. speed trade-offs
- **Performance monitoring**: Track training efficiency

## GPU/CUDA Support

### Prerequisites

To use GPU acceleration, you need Tesseract compiled with CUDA support. See [Building Tesseract with CUDA](#building-tesseract-with-cuda) below.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | 0 | Enable GPU acceleration (set to 1 to enable) |
| `GPU_DEVICE` | 0 | GPU device ID for multi-GPU systems |
| `CUDA_VISIBLE_DEVICES` | Auto-set | Controls which GPUs are visible to the training process |

### Basic GPU Usage

```bash
# Train with GPU acceleration
make training MODEL_NAME=mymodel USE_GPU=1

# Use specific GPU device (for multi-GPU systems)
make training MODEL_NAME=mymodel USE_GPU=1 GPU_DEVICE=1

# Fine-tune with GPU
make training MODEL_NAME=mymodel START_MODEL=eng USE_GPU=1
```

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

## Building Tesseract with CUDA

### Option 1: Build from Source with CUDA

To enable CUDA support in Tesseract, you need to build it from source with specific flags:

```bash
# Install CUDA toolkit first (version 11.0 or later recommended)
# Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# Clone Tesseract repository
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract

# Configure with CUDA support
./autogen.sh
./configure --enable-cuda

# Build and install
make -j$(nproc)
sudo make install
```

### Option 2: Using Pre-built CUDA-enabled Tesseract

Some distributions provide CUDA-enabled builds. Check your package manager or Tesseract releases.

### Verify CUDA Support

After building, verify CUDA support:

```bash
# Check if CUDA is available
lstmtraining --help | grep -i cuda

# Test GPU acceleration
CUDA_VISIBLE_DEVICES=0 lstmtraining --version
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

### GPU Not Detected

**Problem**: Training runs but doesn't use GPU

**Solutions**:
1. Verify Tesseract is built with CUDA: `lstmtraining --help | grep -i cuda`
2. Check CUDA installation: `nvidia-smi`
3. Verify CUDA_VISIBLE_DEVICES is set correctly
4. Check GPU memory availability

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
