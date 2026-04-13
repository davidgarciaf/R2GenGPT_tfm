#!/bin/bash

# Esta es una plantilla de envío para ejecutar un entrenamiento ligero
# en la cola `tfm.q@deimos`. Ajusta los parámetros según necesites.
# Uso:
#   qsub -q tfm.q@deimos submit_light_train_cluster.sh

## ── parámetros de SGE ─────────────────────────────────────────────
#$ -cwd                  # trabajar en el directorio actual
#$ -V                    # exportar variables de entorno
#$ -j y                 # juntar stdout y stderr
#$ -o logs/light_train.log
## solicitar memoria de host suficiente para carga de datos y modelo
#$ -l h_vmem=24G
## puedes cambiar la cola si tu cluster tiene varias disponibles
##$ -q tfm.q@deimos

## activar entorno virtual
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "[Light train] nodo: $(hostname)"
echo "GPU disponible: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /mnt/sd5/users/dgarcia/R2GenGPT

# Ejecutar entrenamiento con pocos batches y una época para validación rápida
python train.py \
    --dataset iu_xray \
    --annotation /mnt/sd5/users/dgarcia/data/iu_xray/annotation.json \
    --base_dir /mnt/sd5/users/dgarcia/data/iu_xray/images \
    --save_dir save/iu_xray/v1_light \
    --max_epochs 1 \
    --batch_size 2 \
    --val_batch_size 2 \
    --limit_train_batches 0.01 \
    --limit_val_batches 0.01 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    --strategy ddp

echo "[Light train] finalizado"
