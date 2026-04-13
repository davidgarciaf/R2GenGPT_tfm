#!/bin/bash

dataset="iu_xray"
annotation="data/iu_xray/annotation.json"
base_dir="./data/iu_xray/images"
# by default try to pick the most recent checkpoint from the save directory;
# users may still override by exporting DELTA_FILE or editing this variable.
savepath="./save/$dataset/$version"

# if delta_file is not defined externally, find latest checkpoint
if [ -z "${delta_file}" ]; then
  if compgen -G "${savepath}/checkpoints/*.pth" > /dev/null; then
    delta_file=$(ls -t ${savepath}/checkpoints/*.pth | head -n1)
    echo "Using checkpoint: $delta_file"
  else
    echo "ERROR: no checkpoint found in ${savepath}/checkpoints"
    exit 1
  fi
fi

version="v1_shallow"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

quant_opts="--load_in_4bit True \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_use_double_quant False \
  --bnb_4bit_quant_type nf4"

python -u train.py \
  --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    ${quant_opts} \
    2>&1 |tee -a ${savepath}/log.txt
