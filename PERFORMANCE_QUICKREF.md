# Performance Optimization Quick Reference

## GPU/CUDA Training (GPU-Only Mode)

```bash
# Enable GPU (forces GPU-only, prevents CPU fallback)
make training MODEL_NAME=mymodel USE_GPU=1

# Select specific GPU device
make training MODEL_NAME=mymodel USE_GPU=1 GPU_DEVICE=1
```

**Important**: `USE_GPU=1` forces GPU-only mode by setting `OMP_THREAD_LIMIT=1`. Training will fail if GPU is not available (no CPU fallback).

## CPU Optimization

```bash
# Use all available CPU cores
make training MODEL_NAME=mymodel OMP_NUM_THREADS=$(nproc)

# Use specific number of threads
make training MODEL_NAME=mymodel OMP_NUM_THREADS=8
```

## Performance Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_GPU` | 0 | 1=GPU-only mode (OMP_THREAD_LIMIT=1), 0=CPU only |
| `GPU_DEVICE` | 0 | GPU device ID (0, 1, 2, ...) |
| `OMP_NUM_THREADS` | Auto | CPU thread count (ignored when USE_GPU=1) |

## Typical Speedups

- **CPU Multi-threading**: 3-5x faster
- **GPU (CUDA)**: 10-20x faster
- **GPU + Optimizations**: 15-30x faster

## More Information

- Full guide: [CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md)
- Examples: [TRAINING_EXAMPLES.md](./TRAINING_EXAMPLES.md)
- General usage: [README.md](./README.md)
