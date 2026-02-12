# GPU/CPU Acceleration Guide for Tesstrain

This document provides comprehensive information about GPU and CPU acceleration support in tesstrain.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Configuration Options](#configuration-options)
5. [Platform-Specific Setup](#platform-specific-setup)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Overview

Tesstrain now supports hardware acceleration for training Tesseract LSTM models:

- **CPU Parallelization (OpenMP)**: Utilize multiple CPU cores during training
- **GPU Acceleration (OpenCL)**: Leverage GPU compute power for faster training

### Benefits

- **Faster Training**: Reduce training time from days to hours
- **Better Resource Utilization**: Make use of available hardware
- **Flexibility**: Works with NVIDIA, AMD, and Intel GPUs

## Requirements

### For CPU Parallelization

- Multi-core CPU
- Tesseract 5.x with training tools
- Sufficient RAM (2-4GB per thread recommended)

### For GPU Acceleration

#### Software Requirements
1. **Tesseract** built with OpenCL support
   - Compile flag: `-DENABLE_OPENCL=ON`
   
2. **OpenCL Runtime** for your GPU vendor:
   - **NVIDIA**: CUDA Toolkit (includes OpenCL)
   - **AMD**: ROCm or AMD APP SDK
   - **Intel**: Intel OpenCL Runtime

#### Hardware Requirements
- OpenCL 1.2 compatible GPU or later
- Minimum 2GB VRAM recommended
- PCIe x16 slot for discrete GPUs

#### Verification Tools
- `clinfo`: Check OpenCL installation and devices
- `nvidia-smi`: Monitor NVIDIA GPU usage
- `rocm-smi` or `radeontop`: Monitor AMD GPU usage

## Quick Start

### 1. Check Acceleration Support

```bash
./check_acceleration.sh
```

This script will:
- Detect if lstmtraining supports OpenMP
- Check for OpenCL devices
- Provide recommendations for your system

### 2. Basic Usage

#### CPU-Only (OpenMP)
```bash
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=8
```

#### GPU Acceleration (OpenCL)
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes
```

#### Combined CPU + GPU
```bash
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=4 USE_OPENCL=yes
```

### 3. Using Example Configurations

```bash
# For NVIDIA GPUs
make training -f examples/nvidia_gpu_config.mk

# For AMD GPUs  
make training -f examples/amd_gpu_config.mk

# For CPU-only optimization
make training -f examples/cpu_optimized_config.mk
```

## Configuration Options

### OPENMP_THREAD_COUNT

Controls the number of CPU threads used during training.

- **Default**: `0` (auto-detect based on CPU cores)
- **Range**: 1 to number of CPU cores
- **Recommendation**: Start with `$(nproc)` and adjust based on memory usage

**Examples:**
```bash
# Use all CPU cores
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=$(nproc)

# Use specific number of threads
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=8

# Let system auto-detect (default)
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=0
```

### USE_OPENCL

Enables OpenCL for GPU acceleration.

- **Default**: empty (disabled)
- **Values**: `yes`, `1`, or empty
- **Requirement**: Tesseract must be built with OpenCL support

**Examples:**
```bash
# Enable OpenCL
make training MODEL_NAME=mymodel USE_OPENCL=yes

# Disable OpenCL (default)
make training MODEL_NAME=mymodel USE_OPENCL=
```

## Platform-Specific Setup

### NVIDIA GPUs

#### 1. Install CUDA Toolkit
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# Verify installation
nvidia-smi
```

#### 2. Build Tesseract with OpenCL
```bash
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
mkdir build && cd build
cmake .. -DENABLE_OPENCL=ON
make -j$(nproc)
sudo make install
```

#### 3. Verify OpenCL
```bash
clinfo | grep -i nvidia
```

#### 4. Train with GPU
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

#### 5. Monitor GPU Usage
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### AMD GPUs

#### 1. Install ROCm
```bash
# Ubuntu 20.04/22.04
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_*_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo amdgpu-install --usecase=rocm

# Verify installation
rocm-smi
```

#### 2. Build Tesseract with OpenCL
```bash
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
mkdir build && cd build
cmake .. -DENABLE_OPENCL=ON
make -j$(nproc)
sudo make install
```

#### 3. Verify OpenCL
```bash
clinfo | grep -i amd
```

#### 4. Train with GPU
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

### Intel GPUs

#### 1. Install Intel OpenCL Runtime
```bash
# Ubuntu/Debian
sudo apt-get install intel-opencl-icd

# Or download from Intel's website
# https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html
```

#### 2. Build Tesseract with OpenCL
```bash
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
mkdir build && cd build
cmake .. -DENABLE_OPENCL=ON
make -j$(nproc)
sudo make install
```

#### 3. Train with GPU
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes
```

## Performance Tuning

### Finding Optimal Thread Count

1. **Start with CPU count:**
   ```bash
   make training MODEL_NAME=test OPENMP_THREAD_COUNT=$(nproc)
   ```

2. **Monitor memory usage:**
   ```bash
   htop
   ```

3. **If swapping occurs, reduce threads:**
   ```bash
   make training MODEL_NAME=test OPENMP_THREAD_COUNT=4
   ```

### GPU Performance Tips

1. **Monitor GPU utilization:**
   - Should be 80-100% for optimal performance
   - If low, check CPU bottleneck

2. **Adjust CPU threads with GPU:**
   - Use 2-4 CPU threads with GPU enabled
   - More threads may cause CPU bottleneck

3. **Check VRAM usage:**
   - Ensure sufficient GPU memory
   - Reduce batch size if needed

### Benchmarking

```bash
# CPU-only baseline
time make training MODEL_NAME=test OPENMP_THREAD_COUNT=1

# Multi-threaded CPU
time make training MODEL_NAME=test OPENMP_THREAD_COUNT=8

# GPU accelerated
time make training MODEL_NAME=test USE_OPENCL=yes

# Combined
time make training MODEL_NAME=test USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

## Troubleshooting

### lstmtraining not found

**Problem:** The script reports lstmtraining is not in PATH

**Solution:**
```bash
# Check if installed
which lstmtraining

# Add to PATH if needed
export PATH=/path/to/tesseract/training:$PATH

# Or install Tesseract training tools
sudo apt-get install tesseract-ocr tesseract-ocr-all
```

### OpenCL not available

**Problem:** GPU not detected or OpenCL disabled

**Solutions:**

1. **Check if Tesseract has OpenCL:**
   ```bash
   ldd $(which lstmtraining) | grep OpenCL
   ```

2. **Install OpenCL runtime:**
   ```bash
   # NVIDIA
   sudo apt-get install nvidia-opencl-dev
   
   # AMD
   sudo apt-get install rocm-opencl-dev
   
   # Intel
   sudo apt-get install intel-opencl-icd
   ```

3. **Rebuild Tesseract with OpenCL:**
   ```bash
   cmake .. -DENABLE_OPENCL=ON
   make -j$(nproc)
   sudo make install
   ```

### No OpenCL Devices Found

**Problem:** clinfo shows no devices

**Solutions:**

1. **Check driver installation:**
   ```bash
   # NVIDIA
   nvidia-smi
   
   # AMD
   rocm-smi
   ```

2. **Verify GPU is recognized:**
   ```bash
   lspci | grep -i vga
   ```

3. **Check permissions:**
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

### Poor GPU Performance

**Problem:** Training slower with GPU than CPU

**Possible causes:**

1. **CPU bottleneck:** Reduce OPENMP_THREAD_COUNT
2. **Small dataset:** GPU overhead not worth it
3. **Old GPU:** Limited compute capability
4. **Memory transfer:** Data copying overhead

**Solutions:**
- Monitor with `nvidia-smi` or equivalent
- Try different thread counts
- Ensure driver is up to date
- Check thermal throttling

### Out of Memory

**Problem:** GPU or system runs out of memory

**Solutions:**

1. **For CPU (RAM):**
   - Reduce OPENMP_THREAD_COUNT
   - Close other applications
   - Add swap space

2. **For GPU (VRAM):**
   - Check VRAM with `nvidia-smi`
   - Use GPU with more memory
   - Process smaller batches

## FAQ

### Q: Can I use multiple GPUs?

A: OpenCL typically uses the default device. For multi-GPU, you may need to set the device using environment variables specific to your OpenCL implementation.

### Q: Is CUDA support different from OpenCL?

A: CUDA is NVIDIA-specific. Tesseract uses OpenCL for broader compatibility. CUDA Toolkit includes OpenCL runtime.

### Q: How much speedup can I expect?

A: Varies by:
- Dataset size
- GPU model
- CPU cores
- Typically 2-5x with GPU, 2-8x with multi-threaded CPU

### Q: Do I need a high-end GPU?

A: No. Even mid-range GPUs provide significant speedup. Requirements:
- OpenCL 1.2+ support
- 2GB+ VRAM recommended
- Any NVIDIA GTX/RTX, AMD RX, or Intel integrated GPU

### Q: Can I use CPU and GPU together?

A: Yes! Use both:
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

### Q: Does this work on Windows/macOS?

A: Yes, but:
- Windows: Requires OpenCL runtime for your GPU
- macOS: OpenCL built-in but deprecated (Metal preferred in future)
- Linux: Best support for all GPU vendors

### Q: How do I verify acceleration is working?

A: Monitor resource usage:
- CPU: `htop` or `top`
- NVIDIA GPU: `nvidia-smi -l 1`
- AMD GPU: `radeontop` or `rocm-smi`
- Check training log for speed improvements

## Additional Resources

- [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- [OpenCL Introduction](https://www.khronos.org/opencl/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [AMD ROCm](https://rocmdocs.amd.com/)
- [Intel OpenCL](https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html)

## Getting Help

If you encounter issues:

1. Run `./check_acceleration.sh` and share output
2. Check GitHub issues: https://github.com/tesseract-ocr/tesstrain/issues
3. Tesseract forum: https://groups.google.com/g/tesseract-ocr
4. Include:
   - Output of `check_acceleration.sh`
   - GPU model and driver version
   - Tesseract version
   - Operating system
