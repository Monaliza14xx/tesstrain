# GPU/CPU Acceleration Examples

This directory contains example configuration files for different hardware acceleration scenarios.

## Available Examples

### 1. NVIDIA GPU Configuration (`nvidia_gpu_config.mk`)

For systems with NVIDIA GPUs using CUDA/OpenCL.

**Usage:**
```bash
make training -f examples/nvidia_gpu_config.mk
```

Or include settings in your make command:
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

**Requirements:**
- NVIDIA GPU (GTX, RTX, Quadro, Tesla)
- CUDA Toolkit installed
- Tesseract built with OpenCL support

### 2. AMD GPU Configuration (`amd_gpu_config.mk`)

For systems with AMD GPUs using ROCm or OpenCL.

**Usage:**
```bash
make training -f examples/amd_gpu_config.mk
```

**Requirements:**
- AMD GPU (RX, Radeon VII, MI series)
- ROCm or AMD APP SDK installed
- Tesseract built with OpenCL support

### 3. CPU-Optimized Configuration (`cpu_optimized_config.mk`)

For maximum CPU performance without GPU.

**Usage:**
```bash
make training -f examples/cpu_optimized_config.mk
```

Or specify thread count directly:
```bash
make training MODEL_NAME=mymodel OPENMP_THREAD_COUNT=$(nproc)
```

**Requirements:**
- Multi-core CPU
- Sufficient RAM (2-4GB per thread recommended)

## Customizing Configurations

You can copy any example file and modify it for your needs:

```bash
cp examples/nvidia_gpu_config.mk my_custom_config.mk
# Edit my_custom_config.mk with your settings
make training -f my_custom_config.mk
```

## Checking Acceleration Support

Before using GPU acceleration, run the detection script:

```bash
./check_acceleration.sh
```

This will show:
- Available OpenCL devices
- OpenMP thread support
- Recommendations for your system

## Performance Tips

### GPU Acceleration
- **Monitor usage**: Use `nvidia-smi`, `rocm-smi`, or `radeontop`
- **Driver updates**: Keep GPU drivers up to date
- **Memory**: Ensure GPU has sufficient VRAM
- **Multiple GPUs**: OpenCL typically uses the default device

### CPU Optimization
- **Thread count**: Start with `$(nproc)` and adjust based on memory
- **Memory**: Monitor with `htop` - reduce threads if swapping occurs
- **Thermal**: Ensure adequate cooling for sustained training
- **Affinity**: Consider CPU affinity settings for NUMA systems

### Combined CPU + GPU
```bash
make training MODEL_NAME=mymodel USE_OPENCL=yes OPENMP_THREAD_COUNT=4
```

Use moderate CPU threads (4-8) to handle data loading while GPU handles computation.

## Troubleshooting

### GPU Not Detected
1. Run `./check_acceleration.sh`
2. Install appropriate OpenCL runtime
3. Verify with `clinfo` command
4. Rebuild Tesseract with OpenCL support if needed

### Poor Performance
1. Check if GPU is actually being used (monitor tools)
2. Ensure drivers are current
3. Try different OPENMP_THREAD_COUNT values
4. Monitor for thermal throttling

### Build Issues
If Tesseract lacks OpenCL support:
1. Install OpenCL SDK (CUDA, ROCm, or Intel)
2. Rebuild Tesseract with CMake flag: `-DENABLE_OPENCL=ON`
3. Verify with: `ldd $(which lstmtraining) | grep OpenCL`

## See Also

- Main README: [../README.md](../README.md)
- Tesseract Documentation: https://tesseract-ocr.github.io/tessdoc/
- OpenCL Information: https://www.khronos.org/opencl/
