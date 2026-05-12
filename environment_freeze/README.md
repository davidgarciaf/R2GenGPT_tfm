# Environment Freeze - R2GenGPT Llama3.2-1B-Instruct Cuantizado

## Latest Freezes

| Date | Status | Packages | Metadata | Notes |
|------|--------|----------|----------|-------|
| **May 12, 2026** | ✅ Verified | 127 | [Training Run](dgarcia_r2gengpt_training_run_20260512.md) | Used for successful Job 22522 training |
| May 9, 2026 | ✅ Original | 129 | [Metadata](dgarcia_r2gengpt_llama3.2_1b_freeze_metadata.md) | Initial freeze |

---

## Overview

This directory contains a complete freeze/snapshot of the R2GenGPT environment with Llama3.2-1B-Instruct model support. It enables **100% reproducible environment setup** on any Linux machine with NVIDIA GPU and CUDA 11.8.

## Files in This Freeze

| File | Purpose | Date | Size |
|------|---------|------|------|
| `dgarcia_r2gengpt_training_run_20260512.md` | Metadata from successful Job 22522 training run | May 12, 2026 | 6 KB |
| `dgarcia_r2gengpt_llama3.2_1b_pip_freeze_20260512.txt` | Pip freeze (127 packages) used for Job 22522 | May 12, 2026 | 5 KB |
| `dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt` | Complete pip freeze (all 129 packages with exact versions) | May 9, 2026 | 5 KB |
| `requirements_core.txt` | Core dependencies only (useful for understanding what's essential) | May 9, 2026 | 1 KB |
| `dgarcia_r2gengpt_llama3.2_1b_freeze_metadata.md` | Detailed metadata about the environment | May 9, 2026 | 8 KB |
| `install_from_freeze.sh` | Automated script to reproduce the environment | May 9, 2026 | 5 KB |
| `README.md` | This file | - | - |

## Quick Start

### Method 1: Automated (Recommended)

```bash
# Make script executable
chmod +x install_from_freeze.sh

# Create environment in a specific location
./install_from_freeze.sh ~/my_r2gengpt_env

# Or specify any path:
./install_from_freeze.sh /mnt/storage/environments/r2gengpt_llama3.2
```

The script will:
1. Create a fresh Python virtual environment
2. Upgrade pip/setuptools/wheel
3. Install PyTorch 2.4.1+cu118 with CUDA support
4. Install all 129 packages from the freeze
5. Verify the installation

### Method 2: Manual Installation (Step-by-step)

```bash
# 1. Create virtual environment
python3 -m venv /path/to/new/env

# 2. Activate
source /path/to/new/env/bin/activate

# 3. Upgrade tools
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch first (important!)
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 5. Install from freeze
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt
```

### Method 3: Core Dependencies Only

```bash
# Create and activate venv (as above)
python3 -m venv /path/to/new/env
source /path/to/new/env/bin/activate

# Install only core packages (faster)
pip install -r requirements_core.txt
```

## System Requirements

### Minimum
- **Python**: 3.8+
- **CUDA**: 11.8 (required for cu118 torch wheels)
- **GPU Memory**: 4 GB (with quantization)
- **Disk**: 5 GB free space

### Recommended
- **Python**: 3.8.10 (tested version)
- **CUDA**: 11.8
- **CUDNN**: 9.1.0.70 (auto-installed with torch)
- **GPU Memory**: 6-8 GB
- **Disk**: 10 GB free space
- **RAM**: 32 GB
- **CPU**: 8+ cores

## Installation Time

| Task | Time |
|------|------|
| Create venv | 1-2 min |
| Upgrade pip/tools | 1 min |
| Install PyTorch | 5-10 min |
| Install remaining 100+ packages | 10-15 min |
| **Total** | **20-30 min** |

## Verification

After installation, verify the environment is correct:

### Quick Check
```bash
source /path/to/env/bin/activate
python -c "
import torch
from transformers import __version__ as tf_version
import bitsandbytes
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ Transformers: {tf_version}')
print(f'✓ Bitsandbytes: {bitsandbytes.__version__}')
"
```

### Comprehensive Test
```bash
python << 'EOF'
import torch
import transformers
import lightning as pl
import bitsandbytes
import peft
import accelerate

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Transformers: {transformers.__version__}")
print(f"Lightning: {pl.__version__}")
print(f"Bitsandbytes: {bitsandbytes.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print("=" * 60)
print("✅ Environment Ready")
EOF
```

## Troubleshooting

### Issue: CUDA Not Available

**Symptom**: `torch.cuda.is_available()` returns False

**Solutions**:
1. Check your GPU with `nvidia-smi`
2. Ensure CUDA 11.8 is installed: `nvidia-smi | grep "CUDA Version"`
3. Reinstall torch for your CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # For CUDA 11.8 (original)
   pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
       --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: Installation Fails

**Symptom**: pip install fails with compilation errors

**Solutions**:
```bash
# Install without cache
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt --no-cache-dir

# Install with verbose output
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt -v

# Install packages sequentially instead of all at once
cat dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt | while read package; do
    pip install "$package"
done
```

### Issue: Out of Memory

**Symptom**: pip install fails with memory errors

**Solutions**:
```bash
# Reduce memory usage during installation
pip install --no-cache-dir -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt

# Install sequentially (uses less RAM)
# See script above
```

### Issue: Python Version Mismatch

**Symptom**: Some packages won't install for your Python version

**Solutions**:
1. The freeze was created with Python 3.8.10
2. Python 3.9 and 3.10 should work
3. For other versions, install core packages only:
   ```bash
   pip install -r requirements_core.txt
   ```

## Updating the Freeze

If you need to update packages and regenerate the freeze:

```bash
# 1. Activate the environment
source /path/to/env/bin/activate

# 2. Update specific packages
pip install --upgrade transformers

# 3. Regenerate the freeze
pip freeze > dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt

# 4. Create new metadata
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
" > freeze_info.txt
```

## Using with R2GenGPT

After setting up the environment from freeze:

```bash
# 1. Activate
source /path/to/env/bin/activate

# 2. Navigate to R2GenGPT
cd /mnt/sd5/users/dgarcia/R2GenGPT

# 3. Run experiments
bash scripts/run_1-1.shallow_run_iuxray_rep.sh
```

## Compatibility Matrix

| Component | Version | Tested |
|-----------|---------|--------|
| Python | 3.8.10 | ✓ Yes |
| Python | 3.9.x | ✓ Likely |
| Python | 3.10.x | ✓ Likely |
| CUDA | 11.8 | ✓ Yes |
| CUDA | 12.1+ | ⚠ Requires torch reinstall |
| OS | Linux x86_64 | ✓ Yes |
| OS | macOS | ✗ Not tested |
| OS | Windows | ✗ Not tested |

## Reproducibility Guarantees

✓ **Exact Package Versions**: All 129 packages pinned to exact versions  
✓ **CUDA Configuration**: Locked to CUDA 11.8 support  
✓ **Python Version**: Tested with 3.8.10  
✓ **Architecture**: x86_64 Linux (NVIDIA GPU)  
✓ **Dependencies**: All transitive dependencies captured

## Limitations

✗ **Platform Specific**: Limited to Linux x86_64 with NVIDIA GPUs  
✗ **CUDA Version**: Requires CUDA 11.8 (or manual torch reinstall)  
✗ **Python Version**: Tested only with 3.8.10 (others untested)  
✗ **Pre-trained Models**: Frozen environment; doesn't include downloaded models  

## Notes

1. **Model Downloads**: The freeze doesn't include pre-downloaded models (Llama3.2-1B, Swin-Base, etc.). These will be downloaded on first use from Hugging Face.

2. **Token File**: Remember to configure your Hugging Face token in `.hf_token_llama3.env`

3. **Disk Space**: The environment is ~4.6 GB; models add another ~5-10 GB.

4. **CUDA Toolkit**: The freeze includes CUDA libraries via torch wheels, but the system CUDA toolkit (nvidia-smi) must match or be newer than 11.8.

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Bitsandbytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [PEFT (LoRA)](https://github.com/huggingface/peft)

## Support

For issues specific to R2GenGPT, see:
- `/mnt/sd5/users/dgarcia/R2GenGPT/SETUP_GUIDE.md`
- `/mnt/sd5/users/dgarcia/R2GenGPT/ENVIRONMENT_SETUP_SUMMARY.md`

---

**Created**: May 9, 2026  
**Environment**: Llama3.2-1B-Instruct Cuantizado  
**Author**: dgarcia  
**Project**: R2GenGPT - Automatic Radiology Report Generation
