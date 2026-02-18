# Using BATCH_SIZE Parameter with tesstrain

This guide shows how to use the new `BATCH_SIZE` parameter for explicit batch processing with Tesseract 5.x and newer.

## Problem Statement Example

The user wanted to support this lstmtraining command:

```bash
# GPU training with optimized batch size
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

lstmtraining \
  --continue_from eng.traineddata \
  --traineddata output/custom.traineddata \
  --model_output output/checkpoints/custom \
  --train_listfile train.list \
  --max_iterations 10000 \
  --learning_rate 0.0001 \
  --max_image_MB 8000 \
  --batch_size 500 \
  2>&1 | tee training.log
```

## Solution: Using BATCH_SIZE with make training

Now you can achieve the same with tesstrain's Makefile:

```bash
make training MODEL_NAME=custom \
  USE_GPU=1 \
  START_MODEL=eng \
  BATCH_SIZE=500 \
  GPU_MAX_MEMORY=8000 \
  MAX_ITERATIONS=10000 \
  LEARNING_RATE=0.0001 \
  GROUND_TRUTH_DIR=./ground-truth
```

## What This Does

When `USE_GPU=1` and `BATCH_SIZE=500`, the Makefile automatically:

1. **Sets environment variables**:
   - `TESSERACT_OPENCL_DEVICE=GPU:0` - Tells Tesseract to use GPU
   - `OMP_THREAD_LIMIT=1` - Prevents CPU fallback
   - `CUDA_VISIBLE_DEVICES=0` - Makes GPU visible

2. **Passes parameters to lstmtraining**:
   - `--batch_size 500` - Explicit batch size
   - `--max_image_MB 8000` - GPU memory limit
   - `--learning_rate 0.0001` - Learning rate
   - `--max_iterations 10000` - Max iterations
   - Plus all the standard parameters

## Complete Example

```bash
# Step 1: Prepare your training data
mkdir -p data/custom-ground-truth
# ... add your .gt.txt files ...

# Step 2: Run training with explicit batch size
make training MODEL_NAME=custom \
  USE_GPU=1 \
  START_MODEL=eng \
  TESSDATA=/usr/share/tessdata \
  DATA_DIR=./data \
  GROUND_TRUTH_DIR=./data/custom-ground-truth \
  BATCH_SIZE=500 \
  GPU_MAX_MEMORY=8000 \
  MAX_ITERATIONS=10000 \
  LEARNING_RATE=0.0001 \
  DEBUG_INTERVAL=1000
```

## Output Command

The Makefile will generate and execute:

```bash
OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=0 TESSERACT_OPENCL_DEVICE=GPU:0 \
lstmtraining \
  --debug_interval 1000 \
  --traineddata data/custom/custom.traineddata \
  --old_traineddata /usr/share/tessdata/eng.traineddata \
  --continue_from data/eng/custom.lstm \
  --learning_rate 0.0001 \
  --model_output data/custom/checkpoints/custom \
  --train_listfile data/custom/list.train \
  --eval_listfile data/custom/list.eval \
  --max_iterations 10000 \
  --target_error_rate 0.01 \
  --max_image_MB 8000 \
  --net_mode 1 \
  --weight_range 0.1 \
  --momentum 0.9 \
  --batch_size 500 \
2>&1 | tee -a data/custom/training.log
```

## Important Notes

### Tesseract Version Requirement

The `--batch_size` parameter requires **Tesseract 5.x or newer**. To check your version:

```bash
tesseract --version
lstmtraining --help 2>&1 | grep -i batch
```

If `--batch_size` is not supported, leave `BATCH_SIZE=0` (default) and use `GPU_MAX_MEMORY` for implicit batching.

### Choosing Between BATCH_SIZE and GPU_MAX_MEMORY

**Use BATCH_SIZE when**:
- You have Tesseract 5.x or newer
- You want explicit control over batch size
- You're experimenting with different batch sizes

**Use GPU_MAX_MEMORY (implicit batching) when**:
- You have older Tesseract versions
- You want automatic batch sizing
- You want maximum GPU utilization (recommended for most users)

### Recommended Settings by GPU

| GPU | GPU_MAX_MEMORY | BATCH_SIZE (if using) |
|-----|----------------|----------------------|
| T4 (16GB) | 14000 | 300-500 |
| V100 (32GB) | 28000 | 500-1000 |
| A100 (40GB) | 36000 | 1000-2000 |

## Troubleshooting

### "Unknown flag: --batch_size"

Your Tesseract version doesn't support `--batch_size`. Set `BATCH_SIZE=0` (or omit it):

```bash
make training MODEL_NAME=custom \
  USE_GPU=1 \
  GPU_MAX_MEMORY=8000 \
  MAX_ITERATIONS=10000
```

### GPU Still Not Being Used

See [GPU_TROUBLESHOOTING.md](./GPU_TROUBLESHOOTING.md) for complete troubleshooting steps.

### Out of Memory Errors

Reduce batch size or GPU memory:

```bash
make training MODEL_NAME=custom \
  USE_GPU=1 \
  BATCH_SIZE=250 \
  GPU_MAX_MEMORY=6000
```

## Monitoring Training

Monitor GPU utilization:

```bash
watch -n 1 nvidia-smi
```

Expected with proper GPU usage:
- GPU Memory: 6000-14000 MB (depending on settings)
- GPU Utilization: 70-100%
- Process: lstmtraining visible in GPU process list

## See Also

- [CUDA_OPTIMIZATION.md](./CUDA_OPTIMIZATION.md) - Complete GPU optimization guide
- [BATCH_PROCESSING_GUIDE.md](./BATCH_PROCESSING_GUIDE.md) - Batch processing details
- [GPU_TROUBLESHOOTING.md](./GPU_TROUBLESHOOTING.md) - GPU troubleshooting
- [README.md](./README.md) - Main documentation
