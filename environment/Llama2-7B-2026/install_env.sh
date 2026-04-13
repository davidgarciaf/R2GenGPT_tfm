#!/bin/bash

# Script para crear el entorno virtual con soporte para Llama2-7B cuantizado
# Este script gestiona las dependencias y resuelve conflictos

VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "===== Instalación de Entorno Llama2-7B Cuantizado ====="
echo ""

# Activar entorno virtual
echo "[1/5] Activando entorno virtual..."
source "$VENV_PATH/bin/activate"

# Actualizar herramientas básicas
echo "[2/5] Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel

# Instalar PyTorch con CUDA 11.8 (necesario para cuantización)
echo "[3/5] Instalando PyTorch con CUDA 11.8 (esto puede tomar unos minutos)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias core (en orden de importancia)
echo "[4/5] Instalando dependencias del proyecto..."
pip install transformers==4.30.2
pip install peft
pip install lightning==2.0.5
pip install tensorboardX
pip install Pillow
pip install numpy
pip install gradio

# Instalar librerías adicionales para cuantización
echo "[5/5] Instalando librerías para cuantización (bitsandbytes, etc)..."
pip install bitsandbytes
pip install accelerate

echo ""
echo "===== Instalación completada ====="
echo ""
echo "Para activar el entorno en el futuro, usa:"
echo "source $VENV_PATH/bin/activate"
echo ""
