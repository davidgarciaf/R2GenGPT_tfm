#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/tokenizer_cluster.log

# Script para test de tokenizer (ligero)
# Uso: qsub -q student.q@pcgtx1080 submit_tokenizer_cluster.sh

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "=============================================================================="
echo "  TEST DE TOKENIZER EN CLUSTER"
echo "=============================================================================="
echo ""
echo "Nodo: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

# Test de tokenizer (muy ligero, sin cargar modelo)
python3 test_tokenizer_only.py

echo ""
echo "=============================================================================="
echo "  TEST COMPLETADO"
echo "=============================================================================="
