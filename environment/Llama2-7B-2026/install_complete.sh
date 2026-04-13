#!/bin/bash

# ============================================================
# Script de Instalación: Llama2-7B Cuantizado - Instalación Manual
# ============================================================
# Este script instala el entorno para Llama2-7B con soporte
# para cuantización 4-bit usando bitsandbytes y PEFT LoRA
# ============================================================

set -e  # Salir si hay algún error

VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv"
PROJECT_DIR="/mnt/sd5/users/dgarcia/R2GenGPT"

echo "=========================================="
echo "  INSTALACIÓN: Llama2-7B Cuantizado"
echo "=========================================="
echo ""

# Verificar si el entorno virtual existe
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ El entorno virtual no existe en $VENV_PATH"
    echo "   Creando entorno virtual..."
    python3 -m venv "$VENV_PATH"
fi

# Activar el entorno virtual
echo "✓ Activando entorno virtual..."
source "$VENV_PATH/bin/activate"

# Paso 1: Actualizar herramientas básicas
echo ""
echo "[1/5] Actualizando pip, setuptools y wheel..."
pip install --upgrade pip setuptools wheel
echo "✓ Herramientas actualizadas"

# Paso 2: Instalar PyTorch con CUDA 11.8
echo ""
echo "[2/5] Instalando PyTorch 2.4.1 con CUDA 11.8..."
echo "      (Este paso puede tomar 10-15 minutos)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch instalado"

# Paso 3: Instalar transformers (versión específica por compatibilidad)
echo ""
echo "[3/5] Instalando Transformers 4.30.2..."
pip install transformers==4.30.2
echo "✓ Transformers instalado"

# Paso 4: Instalar dependencias core
echo ""
echo "[4/5] Instalando dependencias del proyecto..."
pip install peft>=0.4.0
pip install lightning==2.0.5
pip install tensorboardX>=2.6
pip install Pillow>=9.0.0
pip install numpy
pip install gradio>=3.40.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.2.0
echo "✓ Dependencias core instaladas"

# Paso 5: Instalar librerías de cuantización (CRÍTICO)
echo ""
echo "[5/5] Instalando librerías de cuantización (4-bit/8-bit)..."
echo "      (bitsandbytes y accelerate)"
pip install bitsandbytes>=0.40.0
pip install accelerate>=0.20.0
echo "✓ Librerías de cuantización instaladas"

echo ""
echo "=========================================="
echo "✅ INSTALACIÓN COMPLETADA"
echo "=========================================="
echo ""
echo "Para activar el entorno en el futuro, ejecuta:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Para verificar que todo está correcto, ejecuta:"
echo "  python3 -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo "  python3 -c \"import transformers; print(f'Transformers: {transformers.__version__}')\""
echo "  python3 -c \"import bitsandbytes; print('bitsandbytes: OK')\""
echo ""
echo "Para usar Llama2-7B cuantizado, asegúrate de tener:"
echo "  - El modelo descargado desde Hugging Face"
echo "  - Permisos de acceso en HuggingFace"
echo "  - Token HuggingFace configurado: huggingface-cli login"
echo ""
