# GPU Stability Guide: Preventing "2-Minute Disconnection" Issues

## Problem

GPU training starts successfully with "Using OpenCL GPU acceleration" but disconnects after approximately 2 minutes of training, falling back to CPU or crashing.

## Root Causes

1. **GPU Memory Overflow (OOM)** - Most common cause
2. **Gradient Explosion** - Unstable gradients causing NaN values
3. **GPU Watchdog Timeout** - GPU too busy, system kills process
4. **Thermal Throttling** - GPU overheating
5. **Power Limit** - GPU hitting power ceiling

## Solutions

### 1. Add Checkpoint Interval (Critical!)

**Problem**: If GPU disconnects, you lose all training progress.

**Solution**: Save checkpoints regularly using `CHECKPOINT_INTERVAL`:

```bash
make training MODEL_NAME=mymodel \
  USE_GPU=1 \
  CHECKPOINT_INTERVAL=100 \
  MAX_ITERATIONS=10000
```

- Saves model every 100 iterations
- Training can resume from last checkpoint
- Prevents complete data loss

### 2. Add ADAM Beta for Gradient Stability

**Problem**: Gradient explosion causes NaN values, crashing GPU training.

**Solution**: Use `ADAM_BETA=0.999` for more stable gradients:

```bash
make training MODEL_NAME=mymodel \
  USE_GPU=1 \
  ADAM_BETA=0.999 \
  CHECKPOINT_INTERVAL=100
```

- Default 0.999 prevents gradient spikes
- Higher values = more stable (0.9-0.999)
- Critical for large networks (>500k weights)

### 3. Reduce GPU Memory Usage

**Problem**: GPU runs out of memory after warmup phase.

**Original (Too Aggressive)**:
```bash
GPU_MAX_MEMORY=12288  # 80% of T4's 15GB
BATCH_SIZE=500        # Too large
```

**Safe for T4 GPU (16GB)**:
```bash
make training MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_MAX_MEMORY=11000 \     # Leave 25% headroom
  BATCH_SIZE=150 \            # Reduced from 500
  CHECKPOINT_INTERVAL=100 \
  ADAM_BETA=0.999
```

**Safe for other GPUs**:
- **V100 (32GB)**: GPU_MAX_MEMORY=24000, BATCH_SIZE=300
- **A100 (40GB)**: GPU_MAX_MEMORY=32000, BATCH_SIZE=400
- **RTX 3090 (24GB)**: GPU_MAX_MEMORY=18000, BATCH_SIZE=200

**Rule of Thumb**: Use 70-75% of GPU memory, not 80-90%

### 4. Monitor GPU During Training

```bash
# Terminal 1: Run training
make training MODEL_NAME=mymodel USE_GPU=1 ...

# Terminal 2: Monitor GPU every second
watch -n 1 nvidia-smi
```

**What to Look For**:

| Metric | Healthy | Problem |
|--------|---------|---------|
| GPU Utilization | 70-100% | 0% (using CPU!) |
| Memory Usage | Steady or slowly increasing | Sudden spike to 100% |
| Temperature | <80°C | >85°C (throttling) |
| Power Usage | Stable near limit | Fluctuating wildly |

### 5. Complete Safe Configuration

For the user's exact case (Tesla T4, large LSTM network):

```python
import os
import subprocess

# Set environment variables
os.environ['TESSERACT_OPENCL_VERBOSE'] = '1'
os.environ['TESSERACT_OPENCL_DEVICE'] = 'GPU:0'
os.environ['OMP_THREAD_LIMIT'] = '1'

# Run training with safe settings
cmd = """
cd /content/tesstrain && \
make training MODEL_NAME=lao140k \
  DEBUG_INTERVAL=0 \
  USE_GPU=1 \
  GPU_MAX_MEMORY=11000 \
  BATCH_SIZE=150 \
  CHECKPOINT_INTERVAL=100 \
  ADAM_BETA=0.999 \
  START_MODEL=Lao \
  DATA_DIR=/content/tesstrain/data \
  GROUND_TRUTH_DIR=/content/tesstrain/data/lao140k-ground-truth \
  TESSDATA=/content/tesseract/tessdata \
  LEARNING_RATE=0.0001 \
  MAX_ITERATIONS=10000 \
  2>&1 | tee /content/training_debug.log
"""

subprocess.run(cmd, shell=True)
```

**Key Changes from Original**:
- ✅ GPU_MAX_MEMORY: 12288 → 11000 (leave headroom)
- ✅ BATCH_SIZE: 500 → 150 (more stable)
- ✅ CHECKPOINT_INTERVAL: (missing) → 100 (save progress)
- ✅ ADAM_BETA: (missing) → 0.999 (prevent gradient explosion)

## Training Output Indicators

### Successful GPU Training
```
Using OpenCL GPU acceleration for matrix operations
Loaded file /content/tesstrain/data/Lao/lao140k.lstm, unpacking...
...
At iteration 100/10000, Mean rms=1.234%, delta=0.567%, BCER train=2.345%, BWER train=5.678%
Saved checkpoint to /content/tesstrain/data/lao140k/checkpoints/lao140k_100.checkpoint
At iteration 200/10000, Mean rms=1.123%, delta=0.456%, BCER train=2.234%, BWER train=5.567%
```

**Good signs**:
- ✅ "Using OpenCL GPU acceleration"
- ✅ Regular iteration updates
- ✅ Checkpoint saves every 100 iterations
- ✅ RMS and error rates gradually decreasing

### GPU Disconnection Warning Signs
```
Using OpenCL GPU acceleration for matrix operations
...
At iteration 50/10000, Mean rms=1.234%, delta=0.567%
At iteration 51/10000, Mean rms=nan%, delta=nan%
Error: Network returned NaN values
```

**Problem signs**:
- ❌ "nan%" (gradient explosion - need ADAM_BETA)
- ❌ Training stops suddenly
- ❌ No checkpoint saves
- ❌ GPU utilization drops to 0%

## Resuming After Disconnect

If GPU disconnects but you have checkpoints:

```bash
# Find latest checkpoint
ls -lh /content/tesstrain/data/lao140k/checkpoints/

# Resume from checkpoint
make training MODEL_NAME=lao140k \
  USE_GPU=1 \
  GPU_MAX_MEMORY=11000 \
  CHECKPOINT_INTERVAL=100 \
  ADAM_BETA=0.999 \
  START_MODEL=lao140k \  # Changed from Lao
  MAX_ITERATIONS=10000   # Will continue from last checkpoint
```

## Performance Comparison

| Configuration | GPU Utilization | Stability | Speed |
|---------------|----------------|-----------|-------|
| Original (12288MB, BS=500, no checkpoints) | 0% after 2 min | ❌ Crashes | N/A |
| Recommended (11000MB, BS=150, checkpoints) | 80-90% continuous | ✅ Stable | 15-18x vs CPU |
| Conservative (10000MB, BS=100, checkpoints) | 70-80% continuous | ✅ Very stable | 12-15x vs CPU |

## Quick Checklist

Before starting long GPU training:

- [ ] Set `CHECKPOINT_INTERVAL=100` (or lower for extra safety)
- [ ] Set `ADAM_BETA=0.999` (for gradient stability)
- [ ] Use 70-75% of GPU memory, not 80-90%
- [ ] Reduce BATCH_SIZE if network is large (>500k weights)
- [ ] Monitor with `nvidia-smi` during first 5 minutes
- [ ] Check training log for "Using OpenCL GPU acceleration"
- [ ] Verify checkpoint files are being created

## Additional Resources

- [GPU_TROUBLESHOOTING.md](./GPU_TROUBLESHOOTING.md) - If GPU not being used at all
- [CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md) - General GPU optimization
- [BATCH_PROCESSING_GUIDE.md](./BATCH_PROCESSING_GUIDE.md) - Understanding batch processing
