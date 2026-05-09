#!/bin/bash
# Activar entorno Conda
source /export/anaconda3/bin/activate dgarcia_tfm_clean

# Ir al directorio del proyecto
cd /mnt/sd5/users/dgarcia/R2GenGPT

# Cargar el token de Hugging Face del archivo privado
TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"

# Cargar el token de Hugging Face del archivo privado
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
# File for checkpoints
delta_file="/mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow/checkpoints/checkpoint_epoch8_step4653_bleu0.102054_cider0.277842.pth"

version="v1_shallow_test"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

# Ajustes para el modelo cuantizado a 4 bits
quant_opts="--load_in_4bit True \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_use_double_quant False \
  --bnb_4bit_quant_type nf4"

python -u /mnt/sd5/users/dgarcia/R2GenGPT/train.py \
  --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 4 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 2 \
    --devices 1 \
    ${quant_opts} \
    2>&1 |tee -a ${savepath}/log.txt
