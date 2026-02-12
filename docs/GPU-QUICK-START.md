# GPU Acceleration Quick Start Guide

This is a quick reference guide for using GPU acceleration with tesstrain. For detailed information, see [GPU-ACCELERATION.md](GPU-ACCELERATION.md).

## Prerequisites

1. **GPU Hardware**: NVIDIA, AMD, or Intel GPU
2. **OpenCL Drivers**: Installed for your GPU
3. **Tesseract**: Compiled with OpenCL support (>= 5.3)

## Quick Setup

### Step 1: Verify OpenCL Installation

```bash
# Check if OpenCL is available
clinfo

# Verify lstmtraining has OpenCL support
lstmtraining --help | grep opencl
```

### Step 2: Enable GPU Acceleration

Simply add `OPENCL_ENABLE=1` to your make command:

```bash
# Basic GPU-accelerated training
make training MODEL_NAME=my-model OPENCL_ENABLE=1

# With custom parameters
make training \
  MODEL_NAME=my-model \
  OPENCL_ENABLE=1 \
  MAX_ITERATIONS=10000 \
  LEARNING_RATE=0.002
```

## Common Commands

### Train from Scratch with GPU

```bash
make training \
  MODEL_NAME=my-language \
  OPENCL_ENABLE=1 \
  MAX_ITERATIONS=50000
```

### Fine-tune Existing Model with GPU

```bash
make training \
  MODEL_NAME=my-custom-model \
  START_MODEL=eng \
  TESSDATA=~/tessdata_best \
  OPENCL_ENABLE=1 \
  MAX_ITERATIONS=10000
```

### Generate Traineddata with GPU

```bash
make traineddata \
  MODEL_NAME=my-model \
  OPENCL_ENABLE=1
```

## Performance Monitoring

### Monitor GPU Usage (NVIDIA)

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log GPU stats during training
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > gpu_stats.csv &

# Start training
make training MODEL_NAME=my-model OPENCL_ENABLE=1
```

### Monitor GPU Usage (AMD)

```bash
# Real-time monitoring
radeontop

# Or use rocm-smi for ROCm-based systems
watch -n 1 rocm-smi
```

## Troubleshooting

### Problem: GPU not being used

**Check**:
```bash
# Verify OpenCL device is detected
clinfo | grep "Device Type"

# Should show: Device Type: CL_DEVICE_TYPE_GPU
```

**Solution**: Install appropriate GPU drivers
```bash
# NVIDIA
sudo apt-get install nvidia-opencl-icd

# AMD
sudo apt-get install mesa-opencl-icd

# Intel
sudo apt-get install intel-opencl-icd
```

### Problem: Out of memory

**Solution**: Reduce model size or batch size
```bash
# Use smaller max iterations
make training MODEL_NAME=my-model OPENCL_ENABLE=1 MAX_ITERATIONS=5000
```

### Problem: Slower than CPU

**Possible causes**:
- Using integrated GPU instead of discrete GPU
- Small model/batch size (GPU overhead not worth it)

**Solution**: Ensure discrete GPU is being used
```bash
# Force GPU device type
export OPENCL_DEVICE_TYPE=GPU

# Check which device is selected during training
lstmtraining --opencl 1 --help
```

## Performance Tips

### 1. Use Larger Iterations for Better GPU Utilization

GPU acceleration shows more benefit with longer training runs:
```bash
# Good GPU utilization
make training MODEL_NAME=my-model OPENCL_ENABLE=1 MAX_ITERATIONS=50000

# Less benefit from GPU
make training MODEL_NAME=my-model OPENCL_ENABLE=1 MAX_ITERATIONS=100
```

### 2. Monitor Efficiency

Compare training speed:
```bash
# CPU training
time make training MODEL_NAME=test-cpu MAX_ITERATIONS=1000

# GPU training
time make training MODEL_NAME=test-gpu MAX_ITERATIONS=1000 OPENCL_ENABLE=1
```

### 3. Single GPU per Training Session

Don't run multiple GPU training jobs simultaneously unless you have multiple GPUs:
```bash
# BAD: Will compete for GPU resources
make training MODEL_NAME=model1 OPENCL_ENABLE=1 &
make training MODEL_NAME=model2 OPENCL_ENABLE=1 &

# GOOD: One at a time
make training MODEL_NAME=model1 OPENCL_ENABLE=1
make training MODEL_NAME=model2 OPENCL_ENABLE=1
```

## Expected Performance

Typical speedup compared to CPU training:

| GPU Type | Speedup | Notes |
|----------|---------|-------|
| High-end GPU (RTX 4090, RX 7900 XTX) | 4-5x | Best for large models |
| Mid-range GPU (RTX 3060, RX 6700 XT) | 2.5-3.5x | Good for most training |
| Low-end/Integrated GPU | 1.2-2x | May not be worth it |

## Quick Reference

| Task | Command |
|------|---------|
| Enable GPU | Add `OPENCL_ENABLE=1` to make command |
| Check GPU status | `nvidia-smi` or `radeontop` |
| Verify OpenCL | `clinfo` |
| Monitor during training | `watch -n 1 nvidia-smi` |
| Disable GPU | Omit `OPENCL_ENABLE=1` or set `OPENCL_ENABLE=0` |

## Next Steps

For more detailed information:
- Read [GPU-ACCELERATION.md](GPU-ACCELERATION.md) for comprehensive guide
- See [TESSERACT-INTERNALS.md](TESSERACT-INTERNALS.md) for technical details
- Check [README.md](../README.md) for general usage

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review [GPU-ACCELERATION.md](GPU-ACCELERATION.md) for detailed solutions
3. Open GPU-related issues on [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract/issues) or tesstrain issues on [tesstrain GitHub](https://github.com/Monaliza14xx/tesstrain/issues)
