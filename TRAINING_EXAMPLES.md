# Example Training Configurations

This file contains example training commands for different scenarios and hardware configurations.

## Basic Training Examples

### 1. Train from Scratch (CPU Only)
```bash
make training \
  MODEL_NAME=mymodel \
  MAX_ITERATIONS=50000 \
  LEARNING_RATE=0.002 \
  OMP_NUM_THREADS=8
```

### 2. Fine-tune Existing Model (CPU Only)
```bash
make training \
  MODEL_NAME=mymodel \
  START_MODEL=eng \
  TESSDATA=~/tessdata_best \
  MAX_ITERATIONS=20000 \
  LEARNING_RATE=0.0001 \
  OMP_NUM_THREADS=8
```

### 3. Train with GPU Acceleration
```bash
make training \
  MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  MAX_ITERATIONS=50000 \
  BATCH_SIZE=200
```

### 4. Fine-tune with GPU + CPU Optimization
```bash
make training \
  MODEL_NAME=mymodel \
  START_MODEL=eng \
  TESSDATA=~/tessdata_best \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  OMP_NUM_THREADS=8 \
  BATCH_SIZE=200 \
  MAX_ITERATIONS=20000 \
  LEARNING_RATE=0.0001
```

## Performance Optimized Examples

### 5. Maximum Performance (High-end GPU)
```bash
make training \
  MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  BATCH_SIZE=300 \
  OMP_NUM_THREADS=16 \
  DEBUG_INTERVAL=1000 \
  MAX_ITERATIONS=100000
```

### 6. Balanced Performance (Mid-range GPU)
```bash
make training \
  MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  BATCH_SIZE=150 \
  OMP_NUM_THREADS=8 \
  MAX_ITERATIONS=50000
```

### 7. Memory Constrained (Low VRAM)
```bash
make training \
  MODEL_NAME=mymodel \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  BATCH_SIZE=50 \
  MAX_ITERATIONS=50000
```

### 8. Multi-GPU Training (Different Models)
```bash
# Terminal 1 - Train model 1 on GPU 0
make training MODEL_NAME=model1 USE_GPU=1 GPU_DEVICE=0 &

# Terminal 2 - Train model 2 on GPU 1
make training MODEL_NAME=model2 USE_GPU=1 GPU_DEVICE=1 &

# Wait for both to complete
wait
```

## Language-Specific Examples

### 9. Train Indic Language Model
```bash
make training \
  MODEL_NAME=myindic \
  LANG_TYPE=Indic \
  USE_GPU=1 \
  MAX_ITERATIONS=50000 \
  LEARNING_RATE=0.002
```

### 10. Train RTL (Right-to-Left) Model
```bash
make training \
  MODEL_NAME=myrtl \
  LANG_TYPE=RTL \
  USE_GPU=1 \
  MAX_ITERATIONS=50000 \
  LEARNING_RATE=0.002
```

## Advanced Training Examples

### 11. Custom Network Architecture
```bash
make training \
  MODEL_NAME=mymodel \
  NET_SPEC='[1,48,0,1 Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256 O1c###]' \
  USE_GPU=1 \
  MAX_ITERATIONS=100000 \
  LEARNING_RATE=0.002
```

### 12. High Quality Training (Lower Error Rate)
```bash
make training \
  MODEL_NAME=highquality \
  USE_GPU=1 \
  MAX_ITERATIONS=200000 \
  TARGET_ERROR_RATE=0.001 \
  LEARNING_RATE=0.0005 \
  DEBUG_INTERVAL=1000
```

### 13. Fast Training (Quick Iterations)
```bash
make training \
  MODEL_NAME=quicktest \
  USE_GPU=1 \
  MAX_ITERATIONS=10000 \
  TARGET_ERROR_RATE=0.05 \
  BATCH_SIZE=200
```

### 14. Training with Epochs Instead of Iterations
```bash
# EPOCHS automatically calculates iterations based on training data size
make training \
  MODEL_NAME=mymodel \
  EPOCHS=50 \
  USE_GPU=1 \
  BATCH_SIZE=200
```

## Testing and Evaluation

### 15. Generate Training Plots
```bash
make plot MODEL_NAME=mymodel
```

### 16. Create Traineddata Files from Checkpoints
```bash
# Create both best (float) and fast (int) models from all checkpoints
make traineddata MODEL_NAME=mymodel

# Create models from specific checkpoints only
make traineddata MODEL_NAME=mymodel \
  CHECKPOINT_FILES="$(ls -t data/mymodel/checkpoints/*.checkpoint | head -5)"
```

### 17. Evaluate Checkpoints
```bash
make evaluation MODEL_NAME=mymodel
```

## Full Training Workflow

### 18. Complete Training Pipeline
```bash
#!/bin/bash
# Complete training workflow with GPU acceleration

MODEL=mymodel
START=eng
TESSDATA_DIR=~/tessdata_best
GROUND_TRUTH=data/${MODEL}-ground-truth

# Step 1: Prepare ground truth data (if needed)
# Place your .png/.tif images and .gt.txt files in ${GROUND_TRUTH}

# Step 2: Download language data (first time only)
make tesseract-langdata

# Step 3: Train the model
make training \
  MODEL_NAME=${MODEL} \
  START_MODEL=${START} \
  TESSDATA=${TESSDATA_DIR} \
  USE_GPU=1 \
  GPU_DEVICE=0 \
  OMP_NUM_THREADS=8 \
  BATCH_SIZE=200 \
  MAX_ITERATIONS=50000 \
  LEARNING_RATE=0.0001 \
  TARGET_ERROR_RATE=0.005 \
  DEBUG_INTERVAL=1000

# Step 4: Generate training plots
make plot MODEL_NAME=${MODEL}

# Step 5: Create final traineddata files
make traineddata MODEL_NAME=${MODEL}

# Step 6: Evaluate all checkpoints
make evaluation MODEL_NAME=${MODEL}

echo "Training complete! Best model is in data/${MODEL}/${MODEL}.traineddata"
echo "Fast models are in data/${MODEL}/tessdata_fast/"
echo "Best models are in data/${MODEL}/tessdata_best/"
```

## Environment Variables Reference

You can also set these as environment variables instead of command-line parameters:

```bash
# Set environment variables
export MODEL_NAME=mymodel
export USE_GPU=1
export GPU_DEVICE=0
export OMP_NUM_THREADS=8
export BATCH_SIZE=200
export MAX_ITERATIONS=50000

# Run training with environment variables
make -e training

# Or combine both approaches
export OMP_NUM_THREADS=16
make training MODEL_NAME=mymodel USE_GPU=1
```

## Tips for Optimal Performance

1. **Start with defaults**: First run with default settings to establish baseline
2. **Monitor GPU usage**: Use `nvidia-smi` to ensure GPU is being utilized
3. **Adjust batch size**: Increase until you run out of GPU memory, then back off 10-20%
4. **Use checkpoints**: Set DEBUG_INTERVAL to 1000-5000 for regular checkpoints
5. **Early stopping**: Set TARGET_ERROR_RATE to stop when accuracy is good enough
6. **Compare speeds**: Time different configurations to find optimal settings
7. **Profile bottlenecks**: Use system tools to identify I/O or CPU bottlenecks

## Common Issues and Solutions

### Issue: Out of GPU Memory
```bash
# Solution: Reduce batch size
make training MODEL_NAME=mymodel USE_GPU=1 BATCH_SIZE=50
```

### Issue: Training Too Slow on CPU
```bash
# Solution: Increase thread count
make training MODEL_NAME=mymodel OMP_NUM_THREADS=$(nproc)
```

### Issue: GPU Not Being Used
```bash
# Solution: Check CUDA installation and set device explicitly
nvidia-smi  # Verify GPU is available
make training MODEL_NAME=mymodel USE_GPU=1 GPU_DEVICE=0
```

### Issue: Want to Resume Training
```bash
# Training automatically resumes from last checkpoint if interrupted
# Just run the same command again
make training MODEL_NAME=mymodel USE_GPU=1
```
