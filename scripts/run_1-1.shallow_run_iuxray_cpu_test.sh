#!/bin/bash
# Script de respaldo: ejecutar en CPU para verificar que el código funciona

# Limpiar entorno CUDA completamente
unset CUDA_VISIBLE_DEVICES
unset CUDA_LAUNCH_BLOCKING
unset CUDA_CACHE_DISABLE
unset CUDA_CACHE_PATH
unset PYTORCH_CUDA_ALLOC_CONF

# Forzar ejecución en CPU
export CUDA_VISIBLE_DEVICES=""
export CUDA_LAUNCH_BLOCKING=1

# Activar entorno Conda
source /export/anaconda3/bin/activate dgarcia_tfm_clean

# Ir al directorio del proyecto
cd /mnt/sd5/users/dgarcia/R2GenGPT

# Cargar el token de Hugging Face del archivo privado
TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"

if [ -f "$TOKEN_FILE" ]; then
    export $(cat $TOKEN_FILE | xargs)
    echo "Token exportado desde el archivo $TOKEN_FILE"
else
    echo "ERROR: No se encontró el archivo $TOKEN_FILE con el token de Hugging Face."
    exit 1
fi

dataset="iu_xray"
annotation="/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json"
base_dir="/mnt/sd5/users/dgarcia/data/iu_xray/images"

version="v1_shallow_cpu_test" # Versión CPU para testing
savepath="/mnt/sd5/users/dgarcia/R2GenGPT/save/$dataset/$version"

# Crear directorio
mkdir -p "$savepath"

echo "=== EJECUCIÓN EN CPU PARA TESTING ==="
echo "Si esto funciona, el problema es específicamente de CUDA/GPU"
echo "Directorio de salida: $savepath"

# Ejecutar con CPU (batch size reducido, 1 epoch para testing rápido)
python -u /mnt/sd5/users/dgarcia/R2GenGPT/train.py \
    --llama_model "meta-llama/Llama-3.2-1B-Instruct" \
    --low_resource True \
    --precision 32 \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 1 \
    --val_batch_size 1 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 0 \
    --devices 1 \
    --accelerator cpu \
    --strategy auto \
    --max_epochs 1 \
    --limit_val_batches 0.1 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    2>&1 | tee -a ${savepath}/log.txt