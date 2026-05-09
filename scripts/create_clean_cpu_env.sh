#!/bin/bash
# Script para crear un entorno conda completamente nuevo y limpio

# Crear nuevo entorno conda con PyTorch CPU-only
echo "Creando nuevo entorno conda 'dgarcia_tfm_clean_cpu'..."
conda create -n dgarcia_tfm_clean_cpu python=3.10 -y

# Activar el nuevo entorno
echo "Activando nuevo entorno..."
source /export/anaconda3/bin/activate dgarcia_tfm_clean_cpu

# Instalar PyTorch CPU-only
echo "Instalando PyTorch CPU-only..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Instalar otras dependencias necesarias
echo "Instalando dependencias adicionales..."
pip install transformers accelerate bitsandbytes scipy scikit-learn
pip install lightning torchmetrics
pip install pillow opencv-python
pip install nltk spacy
pip install pandas numpy matplotlib

# Verificar instalación
echo "Verificando instalación..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
print('✓ PyTorch CPU-only instalado correctamente')
"

echo "Nuevo entorno 'dgarcia_tfm_clean_cpu' creado exitosamente"
echo "Para usar: conda activate dgarcia_tfm_clean_cpu"