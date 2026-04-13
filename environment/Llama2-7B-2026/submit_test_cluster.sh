#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/test_cluster.log

# Script de envío para verificación rápida en cluster
# Uso: qsub -q student.q@pcgtx1080 submit_test_cluster.sh

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "=============================================================================="
echo "  TEST RÁPIDO EN CLUSTER"
echo "=============================================================================="
echo ""
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""

cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

# Test rápido
python3 test_llama2_quantized.py

echo ""
echo "=============================================================================="
echo "  TEST COMPLETADO"
echo "=============================================================================="
