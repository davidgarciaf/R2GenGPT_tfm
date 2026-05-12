#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l h_vmem=16G
#$ -o /mnt/sd5/users/dgarcia/R2GenGPT/logs/run_1-1.shallow_run_iuxray_rep.$JOB_ID.out
#$ -e /mnt/sd5/users/dgarcia/R2GenGPT/logs/run_1-1.shallow_run_iuxray_rep.$JOB_ID.err

set -eo pipefail

echo "=================================================="
echo "R2GenGPT - IU-Xray shallow replication"
echo "JOB_ID=${JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "=================================================="

# ============================================================
# CLEAN TMP
# ============================================================

echo
echo "=== CLEANING OLD TMP FILES ==="

rm -rf /tmp/cuda_libs_*
rm -rf /tmp/hf_*

echo
echo "=== TMP STATUS AFTER CLEANUP ==="

df -h /tmp

# ============================================================
# ULIMIT
# ============================================================

echo
echo "=== ULIMIT ==="

ulimit -a

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR="/mnt/sd5/users/dgarcia/R2GenGPT"

VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv"

TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"

dataset="iu_xray"
annotation="/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json"
base_dir="/mnt/sd5/users/dgarcia/data/iu_xray/images"

version="v1_shallow_rep"
run_date="$(date +%Y%m%d_%H%M%S)"

savepath="/mnt/sd5/users/dgarcia/R2GenGPT/save/${dataset}/${version}/${run_date}"

mkdir -p "${savepath}"
mkdir -p "${PROJECT_DIR}/logs"

echo "=== SAVE PATH ==="
echo "${savepath}"

# ============================================================
# ACTIVATE VENV
# ============================================================

echo
echo "=== ACTIVATING VENV ==="

source "${VENV_PATH}/bin/activate"

echo "which python = $(which python)"
echo "python version = $(python --version)"

# ============================================================
# VERIFY TRANSFORMERS VERSION
# ============================================================

echo
echo "=== VERIFY TRANSFORMERS VERSION ==="

python - <<'PYVER'
import transformers
print("transformers version =", transformers.__version__)

from packaging import version

if version.parse(transformers.__version__) < version.parse("4.45.0"):
    raise RuntimeError(
        "Transformers too old for Llama 3.2. "
        "Install transformers>=4.45"
    )
PYVER

# ============================================================
# NVIDIA CHECK
# ============================================================

echo
echo "=== NVIDIA-SMI ==="

nvidia-smi || {
    echo "ERROR: nvidia-smi failed"
    exit 1
}

# ============================================================
# COPY MINIMAL CUDA LIBS
# ============================================================

echo
echo "=== COPYING MINIMAL CUDA LIBS TO LOCAL TMP ==="

LOCAL_CUDA_LIBS="/tmp/cuda_libs_${JOB_ID:-$$}"

rm -rf "$LOCAL_CUDA_LIBS"
mkdir -p "$LOCAL_CUDA_LIBS"

CUDA_SRC="${VENV_PATH}/lib/python3.8/site-packages"

find "$CUDA_SRC" \( \
    -name "libcublas.so.11" -o \
    -name "libcublasLt.so.11" \
\) -exec cp -L {} "$LOCAL_CUDA_LIBS/" \;

export LD_LIBRARY_PATH="$LOCAL_CUDA_LIBS:${LD_LIBRARY_PATH:-}"

echo
echo "=== LOCAL CUDA LIBS ==="

ls -lh "$LOCAL_CUDA_LIBS"

echo
echo "=== TMP STATUS AFTER CUDA COPY ==="

df -h /tmp

echo
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ============================================================
# PYTORCH CUDA TEST
# ============================================================

echo
echo "=== PYTORCH CUDA TEST ==="

python - <<'PYTEST'
import torch

print("torch version:", torch.__version__)
print("torch path:", torch.__file__)

print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print("gpu count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"gpu {i}: {torch.cuda.get_device_name(i)}")

x = torch.randn(100, 100).cuda()
y = torch.matmul(x, x)

print("basic cuda test OK:", y.shape)

print("CUDA SUCCESS")
PYTEST

# ============================================================
# LOAD HF TOKEN
# ============================================================

echo
echo "=== LOADING HF TOKEN ==="

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: token file not found"
    exit 1
fi

export $(cat "$TOKEN_FILE" | xargs)

echo "HF token loaded"

# ============================================================
# HF CACHE
# ============================================================

cd "$PROJECT_DIR"

export HF_HOME="/mnt/sd5/users/dgarcia/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

mkdir -p "$HF_HOME"

# IMPORTANT FOR LLAMA 3.x
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

# CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================
# QUICK MODE
# ============================================================

if [ "${QUICK:-0}" = "1" ]; then

    echo
    echo "=== QUICK TEST MODE ==="

    bs=2
    vbs=2
    maxep=1
    workers=2
    limval=0.0

else

    echo
    echo "=== FULL TRAINING MODE ==="

    bs=2
    vbs=2
    maxep=15
    workers=1
    limval=1.0

fi

# ============================================================
# QUANTIZATION
# ============================================================

quant_opts="--load_in_4bit True \
--bnb_4bit_compute_dtype float16 \
--bnb_4bit_use_double_quant False \
--bnb_4bit_quant_type nf4"

# ============================================================
# ENVIRONMENT CHECK
# ============================================================

echo
echo "=== ENVIRONMENT CHECK ===" | tee -a "${savepath}/log.txt"

python - <<'PYCHECK' 2>&1 | tee -a "${savepath}/log.txt"
import os
import sys
import torch
import transformers

print("sys.executable =", sys.executable)
print("sys.version =", sys.version)

print("torch =", torch.__version__)
print("torch path =", torch.__file__)

print("transformers =", transformers.__version__)

print("cuda available =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))

print("HF_HOME =", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
print("HUGGINGFACE_HUB_CACHE =", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
PYCHECK

# ============================================================
# TRAINING
# ============================================================

echo
echo "=== START TRAINING ==="

python -u train.py \
    --llama_model "meta-llama/Llama-3.2-1B-Instruct" \
    --low_resource True \
    --precision 16-mixed \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size ${bs} \
    --val_batch_size ${vbs} \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers ${workers} \
    --devices 1 \
    --strategy auto \
    --max_epochs ${maxep} \
    --limit_val_batches ${limval} \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    ${quant_opts} \
    2>&1 | tee -a ${savepath}/log.txt

EXIT_CODE=$?

# ============================================================
# CLEANUP
# ============================================================

echo
echo "=== CLEANING TMP FILES ==="

rm -rf "$LOCAL_CUDA_LIBS"

echo
echo "=== TMP STATUS AFTER CLEANUP ==="

df -h /tmp

echo
echo "=================================================="
echo "TRAINING FINISHED"
echo "EXIT CODE: $EXIT_CODE"
echo "DATE: $(date)"
echo "=================================================="

exit $EXIT_CODE