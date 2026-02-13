# GPU Troubleshooting: "Still Using CPU" Issue

## Problem

You've set `USE_GPU=1` and all environment variables are correct, but training still uses CPU instead of GPU.

## Quick Diagnosis

Run this command to check your setup:
```bash
./src/validate_gpu.sh
```

## Common Causes (in order of likelihood)

### 1. Tesseract Not Built with OpenCL (MOST COMMON)

**How to check**:
```bash
lstmtraining --help 2>&1 | grep -i opencl
```

**If returns nothing**: Tesseract does NOT have OpenCL support. This is the issue!

**Solution**: Rebuild Tesseract with OpenCL support:

```bash
# Install OpenCL development files
sudo apt-get install nvidia-opencl-dev ocl-icd-opencl-dev

# Clone and build Tesseract
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
./autogen.sh
./configure --enable-opencl
make -j$(nproc)
sudo make install
sudo ldconfig

# Verify OpenCL support
lstmtraining --help 2>&1 | grep -i opencl
# Should now show OpenCL-related options
```

### 2. OpenCL Runtime Not Installed

**How to check**:
```bash
clinfo
```

**If command not found**:
```bash
sudo apt-get install clinfo
clinfo
```

**If shows no devices**: Install OpenCL runtime:
```bash
# For NVIDIA GPUs
sudo apt-get install nvidia-opencl-icd-xxx  # where xxx matches your driver version

# Verify
clinfo | grep "Device Type"
# Should show: Device Type = GPU
```

### 3. GPU Drivers Not Installed

**How to check**:
```bash
nvidia-smi
```

**If command not found or fails**: Install NVIDIA drivers:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall
# OR install specific version
sudo apt-get install nvidia-driver-525  # or your preferred version

# Reboot
sudo reboot

# After reboot, verify
nvidia-smi
```

### 4. Wrong GPU Device ID

**How to check**:
```bash
nvidia-smi -L
```

Shows all GPUs. If you have multiple GPUs, ensure `GPU_DEVICE` matches the right one:
```bash
make training MODEL_NAME=test USE_GPU=1 GPU_DEVICE=0  # First GPU
make training MODEL_NAME=test USE_GPU=1 GPU_DEVICE=1  # Second GPU
```

## Verification Steps

After fixing, verify GPU is actually being used:

### Step 1: Check Training Output

When you run training with `USE_GPU=1`, you should see:
```
⚠️  GPU Mode Enabled - Verifying GPU availability...
✓ GPU detected:
Tesla T4, 16130 MiB
```

If you see warnings instead, follow the messages.

### Step 2: Monitor GPU During Training

**Terminal 1** - Start training:
```bash
make training MODEL_NAME=test USE_GPU=1
```

**Terminal 2** - Monitor GPU:
```bash
watch -n 1 nvidia-smi
```

**What to look for**:
- GPU Utilization: Should be 70-100%
- Memory Usage: Should match your GPU_MAX_MEMORY setting (~12-14GB for T4)
- Process: `lstmtraining` should be listed under GPU processes
- Temperature: Will increase to 60-80°C

**If GPU utilization is 0%**: Training is NOT using GPU. Go back to cause #1 (Tesseract not built with OpenCL).

### Step 3: Check lstmtraining Command

The command should include:
```bash
OMP_THREAD_LIMIT=1 CUDA_VISIBLE_DEVICES=0 TESSERACT_OPENCL_DEVICE=GPU:0 \
lstmtraining \
  --max_image_MB 12000 \
  --net_mode 1 \
  ...
```

If these are present but GPU still not used = Tesseract lacks OpenCL support.

## Still Not Working?

### Check Tesseract Version

```bash
tesseract --version
lstmtraining --version
```

Make sure you're using the version you just built, not an old system version:
```bash
which lstmtraining
# Should show /usr/local/bin/lstmtraining (if installed to /usr/local)
```

If it shows `/usr/bin/lstmtraining`, the system version is being used. Fix PATH or remove old version:
```bash
sudo apt-get remove tesseract-ocr
# Then verify
which lstmtraining
```

### Check OpenCL Library

```bash
ldd $(which lstmtraining) | grep -i opencl
```

Should show OpenCL libraries. If not found, Tesseract wasn't properly linked with OpenCL.

### Enable OpenCL Debug

```bash
export OPENCL_VENDOR_PATH=/etc/OpenCL/vendors
export OCL_ICD_FILENAMES=/path/to/your/opencl.so
OPENCL_CACHE_PATH=/tmp lstmtraining --help
```

Check for OpenCL-related error messages.

## Expected Performance

When GPU is working correctly:

| Hardware | Speed vs CPU |
|----------|--------------|
| T4 GPU | 10-20x faster |
| V100 GPU | 15-25x faster |
| A100 GPU | 20-30x faster |

If you're not seeing this speedup, GPU is not being used.

## Summary Checklist

- [ ] Tesseract built with --enable-opencl (`lstmtraining --help | grep opencl`)
- [ ] OpenCL runtime installed (`clinfo` shows GPU)
- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] Correct GPU device selected (`nvidia-smi -L`)
- [ ] GPU utilization >70% during training (`watch nvidia-smi`)
- [ ] Memory usage matches GPU_MAX_MEMORY setting
- [ ] lstmtraining process listed in nvidia-smi
- [ ] Training 10-20x faster than CPU

If all checked and still having issues, the problem may be with your specific GPU or OpenCL installation. Check Tesseract GitHub issues for similar problems.

## Getting Help

When asking for help, provide:

```bash
# 1. Tesseract version and build info
tesseract --version
lstmtraining --help 2>&1 | head -20

# 2. GPU info
nvidia-smi

# 3. OpenCL info
clinfo | grep -A 5 "Platform Name"

# 4. Training command and output
# Include the full make command and first 50 lines of output
```

This information helps diagnose the specific issue.
