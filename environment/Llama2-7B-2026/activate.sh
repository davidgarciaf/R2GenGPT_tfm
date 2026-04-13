#!/bin/bash

# Script de activación rápida del entorno Llama2-7B Cuantizado

VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ El entorno virtual no existe en: $VENV_PATH"
    exit 1
fi

# Activar entorno virtual
source "$VENV_PATH/bin/activate"

# Configurar variables de entorno para optimización CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3000

echo "✅ Entorno Llama2-7B Cuantizado activado"
echo ""
echo "📍 Ruta: $VENV_PATH"
echo "🐍 Python: $(python3 --version)"
echo ""
echo "Librerías cargadas:"
python3 -c "import torch; print(f'  ✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch: Error"
python3 -c "import transformers; print(f'  ✓ Transformers: {transformers.__version__}')" 2>/dev/null || echo "  ✗ Transformers: Error"
python3 -c "import bitsandbytes; print('  ✓ bitsandbytes: OK')" 2>/dev/null || echo "  ✗ bitsandbytes: Error"
python3 -c "import peft; print(f'  ✓ PEFT: {peft.__version__}')" 2>/dev/null || echo "  ✗ PEFT: Error"
python3 -c "import lightning; print(f'  ✓ Lightning: {lightning.__version__}')" 2>/dev/null || echo "  ✗ Lightning: Error"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo ""
    echo "🎮 GPU Disponible:"
    python3 -c "import torch; print(f'  ✓ Device: {torch.cuda.get_device_name(0)}')" 2>/dev/null
    python3 -c "import torch; print(f'  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" 2>/dev/null
fi

echo ""
echo "Listo para usar 🚀"
