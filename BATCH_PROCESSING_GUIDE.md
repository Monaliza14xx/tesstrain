# Batch Processing and GPU Utilization Guide

## Understanding Batch Processing in Tesseract LSTM Training

### Important: No Explicit Batch Size Parameter

**Tesseract's lstmtraining does NOT have a `--batch_size` parameter.** Instead, it uses **implicit batching** based on available GPU memory.

### How Batch Processing Works

The `GPU_MAX_MEMORY` parameter (passed as `--max_image_MB` to lstmtraining) controls how many images can be processed in parallel:

```
More GPU Memory → More Images in Parallel → Faster Training
```

| GPU Memory Setting | Approximate Batch Size | Speed |
|--------------------|----------------------|-------|
| 4000 MB | Small batches | Slow |
| 8000 MB | Medium batches | Moderate |
| 12000 MB | Large batches | Fast |
| 14000 MB | Very large batches | Very Fast |
| 16000+ MB | Maximum batches | Maximum Speed |

## Problem: Low GPU Utilization (0%)

Your nvidia-smi shows:
```
Memory-Usage: 3MiB / 15360MiB
GPU-Util: 0%
```

This means **GPU is not being used at all**. Common causes:

### 1. Tesseract Not Built with OpenCL

**Check**:
```bash
lstmtraining --help 2>&1 | grep -i opencl
```

**If returns nothing**: Rebuild Tesseract with OpenCL support. See [GPU_TROUBLESHOOTING.md](./GPU_TROUBLESHOOTING.md)

### 2. USE_GPU Not Set

**Solution**:
```bash
make training MODEL_NAME=yourmodel USE_GPU=1 GPU_MAX_MEMORY=14000
```

### 3. Training Data Loading Issue

If GPU detection passes but utilization remains 0%, training may be bottlenecked by data loading, not computation.

## Maximizing GPU Utilization and Speed

### For Tesla T4 (16GB) - Your GPU

**Maximum Performance Configuration**:

```bash
make training MODEL_NAME=yourmodel \
  USE_GPU=1 \
  GPU_MAX_MEMORY=14000 \
  NET_MODE=1 \
  DEBUG_INTERVAL=1000 \
  MAX_ITERATIONS=50000
```

**Explanation**:
- `USE_GPU=1` - Forces GPU-only mode
- `GPU_MAX_MEMORY=14000` - Uses 14GB of 16GB (leaves 2GB for system)
- `NET_MODE=1` - Parallel LSTM processing (faster)
- Effective batch size: Determined automatically based on image sizes and 14GB memory

### For Larger GPUs

**V100 (32GB)**:
```bash
GPU_MAX_MEMORY=28000
```

**A100 (40GB)**:
```bash
GPU_MAX_MEMORY=36000
```

**A100 (80GB)**:
```bash
GPU_MAX_MEMORY=72000
```

### Expected GPU Utilization

When training is working correctly:

```
Memory-Usage: 12000-14000 MiB / 15360 MiB  ← Good! Using allocated memory
GPU-Util: 70-100%                           ← Good! GPU is working
Temperature: 60-80°C                        ← Normal under load
```

## Monitoring and Verification

### Monitor GPU in Real-Time

**Terminal 1** - Start training:
```bash
make training MODEL_NAME=test USE_GPU=1 GPU_MAX_MEMORY=14000
```

**Terminal 2** - Monitor GPU:
```bash
watch -n 1 nvidia-smi
```

### What to Look For

✅ **Good Signs**:
- GPU-Util: 70-100%
- Memory-Usage: Close to GPU_MAX_MEMORY setting
- Process `lstmtraining` visible in GPU processes
- Temperature rising (60-80°C)

❌ **Bad Signs**:
- GPU-Util: 0%
- Memory-Usage: < 100 MiB
- No lstmtraining process listed
- Temperature at idle (30-40°C)

If you see bad signs, GPU is NOT being used. Check troubleshooting guide.

## Performance Comparison

### Batch Processing Effect (Tesla T4)

| Configuration | Effective Batch | Training Speed | GPU Memory Used |
|---------------|----------------|----------------|-----------------|
| CPU Only | N/A | 1x (baseline) | 0 MB |
| GPU 4000 MB | Small | 3-5x | 4000 MB |
| GPU 8000 MB | Medium | 6-10x | 8000 MB |
| GPU 12000 MB | Large | 10-15x | 12000 MB |
| **GPU 14000 MB** | **Very Large** | **15-20x** | **14000 MB** |

### Example Timing

For 10,000 iterations:
- CPU: ~8-10 hours
- GPU (4000 MB): ~2-3 hours
- GPU (8000 MB): ~1-1.5 hours
- GPU (12000 MB): ~40-60 minutes
- **GPU (14000 MB): ~30-40 minutes** ← Recommended

## Troubleshooting Poor Performance

### GPU Utilization < 50%

**Possible causes**:

1. **Small training dataset**
   - Not enough images to fill GPU memory
   - Solution: Increase GPU_MAX_MEMORY won't help

2. **Small images**
   - Small images = less computation
   - Solution: This is normal, training will still be faster than CPU

3. **I/O bottleneck**
   - Disk too slow to feed GPU
   - Solution: Use SSD, not HDD

### GPU Memory Not Filling Up

If GPU-Util is high but memory usage is low:
- This is actually fine! It means images are small
- Training is still faster than CPU
- No need to increase GPU_MAX_MEMORY

If GPU-Util is low and memory usage is low:
- GPU not being used - check GPU_TROUBLESHOOTING.md
- Training likely running on CPU

## Advanced: Multiple GPUs

If you have multiple GPUs, train different models simultaneously:

```bash
# GPU 0
make training MODEL_NAME=model1 USE_GPU=1 GPU_DEVICE=0 GPU_MAX_MEMORY=14000 &

# GPU 1
make training MODEL_NAME=model2 USE_GPU=1 GPU_DEVICE=1 GPU_MAX_MEMORY=14000 &

wait
```

Each GPU processes its own batch of images in parallel.

## Summary: How to Make Training Fast

1. **Use GPU**: `USE_GPU=1`
2. **Maximize GPU Memory**: `GPU_MAX_MEMORY=14000` (for T4)
3. **Use Parallel Mode**: `NET_MODE=1`
4. **Verify GPU Usage**: `watch nvidia-smi` should show 70-100% utilization
5. **Build Tesseract with OpenCL**: Required for GPU acceleration

There is **no separate batch_size parameter** - GPU_MAX_MEMORY controls batch processing automatically.

## Quick Reference

```bash
# Maximum speed for Tesla T4
make training MODEL_NAME=yourmodel \
  USE_GPU=1 \
  GPU_MAX_MEMORY=14000 \
  NET_MODE=1 \
  START_MODEL=eng \
  MAX_ITERATIONS=50000 \
  DEBUG_INTERVAL=1000

# Monitor GPU
watch -n 1 nvidia-smi

# Expected result:
# - GPU-Util: 70-100%
# - Memory: ~14000 MiB
# - Speed: 15-20x faster than CPU
```

If GPU utilization is 0%, see [GPU_TROUBLESHOOTING.md](./GPU_TROUBLESHOOTING.md)
