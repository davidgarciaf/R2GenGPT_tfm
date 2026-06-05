#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l h_vmem=32G

set -eo pipefail

echo "=================================================="
echo "R2GenGPT - IU-Xray LoRA"
echo "JOB_ID=${JOB_ID:-manual}"
echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "=================================================="

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR="/mnt/sd5/users/dgarcia/R2GenGPT"

VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv"

TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"

dataset="iu_xray"

annotation="/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json"

base_dir="/mnt/sd5/users/dgarcia/data/iu_xray/images"

version="v1_lora"

run_date="$(date +%Y%m%d_%H%M%S)"

savepath="/mnt/sd5/users/dgarcia/R2GenGPT/save/${dataset}/${version}/${run_date}"

mkdir -p "${savepath}"

# ============================================================
# TMP
# ============================================================

USER_TMP="/mnt/sd5/users/dgarcia/tmp"

mkdir -p "$USER_TMP"

export TMPDIR="${USER_TMP}"

# ============================================================
# ACTIVATE VENV
# ============================================================

echo
echo "=== ACTIVATING VENV ==="

source "${VENV_PATH}/bin/activate"

echo "python = $(which python)"

python --version

# ============================================================
# CUDA ENV
# ============================================================

export LD_LIBRARY_PATH="${VENV_PATH}/lib:${LD_LIBRARY_PATH:-}"

export CUDA_MODULE_LOADING=LAZY

export HF_HUB_ENABLE_HF_TRANSFER=0

export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================
# CUDA TEST
# ============================================================

python - <<'PYTEST'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE")

print("gpu:", torch.cuda.get_device_name(0))
PYTEST

# ============================================================
# HF TOKEN
# ============================================================

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: token file not found"
    exit 1
fi

export $(cat "$TOKEN_FILE" | xargs)

# ============================================================
# HF CACHE
# ============================================================

export HF_HOME="/mnt/sd5/users/dgarcia/.cache/huggingface"

export TRANSFORMERS_CACHE="$HF_HOME"

export HUGGINGFACE_HUB_CACHE="$HF_HOME"

mkdir -p "$HF_HOME"

# ============================================================
# QUICK MODE
# ============================================================

if [ "${QUICK:-0}" = "1" ]; then

    bs=2
    vbs=2
    maxep=1
    workers=2
    limval=0.0
    every_steps=100

else

    bs=2
    vbs=2
    maxep=15
    workers=1
    limval=1.0
    every_steps=10000

fi

# ============================================================
# QLORA
# ============================================================

quant_opts="--load_in_4bit True \
--bnb_4bit_compute_dtype float16 \
--bnb_4bit_use_double_quant False \
--bnb_4bit_quant_type nf4"

# ============================================================
# LORA
# ============================================================

lora_opts=" \
--llm_use_lora True \
--llm_r 16 \
--llm_alpha 32 \
--lora_dropout 0.1"

# ============================================================
# TRAINING
# ============================================================

cd "$PROJECT_DIR"

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
    --learning_rate 5e-5 \
    --gradient_clip_val 1 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers ${workers} \
    --devices 1 \
    --max_epochs ${maxep} \
    --limit_val_batches ${limval} \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    --every_n_train_steps ${every_steps} \
    ${quant_opts} \
    ${lora_opts} \
    2>&1 | tee -a "${savepath}/log.txt"