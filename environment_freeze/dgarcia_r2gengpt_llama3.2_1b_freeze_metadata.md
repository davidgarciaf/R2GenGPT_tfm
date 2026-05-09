# Freeze Metadata - Llama3.2-1B-Instruct Cuantizado
# R2GenGPT Environment Snapshot
# Generated: May 9, 2026

## Environment Information
- **Python Version**: 3.8.10
- **Python Executable**: /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv/bin/python
- **Compiler**: GCC 9.4.0

## PyTorch Configuration
- **PyTorch Version**: 2.4.1+cu118
- **CUDA Version**: 11.8
- **CUDA Available**: Yes
- **Triton Version**: 3.0.0

## Key Dependencies
| Package | Version |
|---------|---------|
| torch | 2.4.1+cu118 |
| torchvision | 0.19.1+cu118 |
| torchaudio | 2.4.1+cu118 |
| transformers | 4.30.2 |
| lightning | 2.0.5 |
| bitsandbytes | 0.45.5 |
| peft | 0.13.2 |
| accelerate | 0.20.0 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.2 |
| gradio | 3.41.2 |
| tensorboardX | 2.6.2.2 |

## Cuantización Configuration
- **Type**: 4-bit NF4
- **Bitsandbytes**: 0.45.5
- **Double Quantization**: False
- **Compute dtype**: float16

## Total Packages
- **Total Installed**: 129 packages
- **Filename**: dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt

## Installation Requirements
- **Python**: 3.8+ (tested with 3.8.10)
- **CUDA**: 11.8 (required for cu118 torch wheels)
- **CUDNN**: 9.1.0.70 (automatically installed with torch)
- **GPU Memory**: Minimum 4 GB (recommended 6-8 GB)
- **Disk Space**: ~5 GB for environment
- **RAM**: 32 GB recommended

## Usage

### Option 1: Create from pip freeze (recommended)
```bash
# Create virtual environment
python3 -m venv /path/to/new/venv

# Activate
source /path/to/new/venv/bin/activate

# Install from freeze
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt
```

### Option 2: Use installation script
```bash
bash install_from_freeze.sh /path/to/new/venv
```

### Option 3: Step-by-step (with proper ordering)
```bash
# 1. Create venv
python3 -m venv /path/to/new/venv
source /path/to/new/venv/bin/activate

# 2. Upgrade base tools
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining packages from freeze
# (This ensures torch is installed first with correct CUDA support)
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt --no-deps
```

## Compatibility

### Tested Platforms
- Linux x86_64 with NVIDIA GPUs
- Tested on: phobos (NVIDIA GPU cluster)

### Untested Platforms
- macOS (GPU libraries might not work)
- Windows (not tested)
- Different CUDA versions (only tested with CUDA 11.8)

## Verification

After installation, verify the environment:

```python
import torch
import transformers
import lightning
import bitsandbytes
import peft
import accelerate

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ Lightning: {lightning.__version__}")
print(f"✓ Bitsandbytes: {bitsandbytes.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
```

## Notes

1. **CUDA 11.8 Required**: The torch wheels are built for CUDA 11.8. If your system has different CUDA, reinstall torch separately.

2. **Python Version**: Created with Python 3.8.10. Python 3.9 or 3.10 should work, but not fully tested.

3. **Reproducibility**: This freeze captures all packages at the exact versions installed. This ensures 100% reproducibility across machines with the same Python version and CUDA 11.8.

4. **Updates**: To update packages, modify and re-run pip install, then regenerate freeze with:
   ```bash
   pip freeze > dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt
   ```

5. **Size**: The installed environment is approximately 4.6 GB.

## Files in This Freeze

- `dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt` - Complete pip freeze output (129 packages)
- `dgarcia_r2gengpt_llama3.2_1b_freeze_metadata.md` - This file
- `install_from_freeze.sh` - Script to reproduce environment
- `requirements_core.txt` - Core dependencies for manual installation

## Troubleshooting

### CUDA Mismatch
If installation fails with CUDA errors:
```bash
# Reinstall torch for your specific CUDA
# Find your CUDA version:
nvidia-smi | grep "CUDA Version"

# Install matching torch (e.g., for CUDA 12.1):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
If pip install runs out of memory:
```bash
# Install with cache disabled (slower but uses less RAM)
pip install --no-cache-dir -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt
```

### Missing Packages
If some packages fail to install:
```bash
# Try installing PyTorch first, then others:
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt --no-deps
```

---

**Freeze Date**: May 9, 2026  
**R2GenGPT Version**: Latest  
**Environment Name**: Llama3.2-1B-Instruct Cuantizado  
**Created by**: dgarcia
