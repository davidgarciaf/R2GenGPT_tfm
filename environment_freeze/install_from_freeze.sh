#!/bin/bash
# Script to reproduce R2GenGPT environment from freeze
# This script creates a new virtual environment with the exact same packages

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  R2GenGPT Environment Reproducor from Freeze                  ║${NC}"
echo -e "${BLUE}║  Llama3.2-1B-Instruct Cuantizado (4-bit NF4)                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get parameters
if [ -z "$1" ]; then
    echo -e "${YELLOW}Uso: $0 /path/to/new/venv${NC}"
    echo ""
    echo "Ejemplos:"
    echo "  $0 ~/my_r2gengpt_env"
    echo "  $0 /mnt/storage/environments/r2gengpt_llama3"
    echo ""
    exit 1
fi

VENV_PATH="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}Configuración:${NC}"
echo "  Directorio venv: $VENV_PATH"
echo "  Directorio freeze: $SCRIPT_DIR"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 no encontrado${NC}"
    echo "Por favor, instala Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Python encontrado: $PYTHON_VERSION${NC}"
echo ""

# Check if venv already exists
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠ El directorio $VENV_PATH ya existe${NC}"
    read -p "¿Deseas eliminarlo y crear uno nuevo? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "Eliminando entorno existente..."
        rm -rf "$VENV_PATH"
    else
        echo "Abortando."
        exit 1
    fi
fi

# Step 1: Create venv
echo -e "${BLUE}[1/5] Creando entorno virtual...${NC}"
python3 -m venv "$VENV_PATH"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Entorno virtual creado${NC}"
else
    echo -e "${RED}✗ Error al crear entorno virtual${NC}"
    exit 1
fi
echo ""

# Activate venv
source "$VENV_PATH/bin/activate"
echo -e "${GREEN}✓ Entorno virtual activado${NC}"
echo ""

# Step 2: Upgrade pip, setuptools, wheel
echo -e "${BLUE}[2/5] Actualizando pip, setuptools y wheel...${NC}"
pip install --upgrade pip setuptools wheel --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Herramientas base actualizadas${NC}"
else
    echo -e "${YELLOW}⚠ Algunos warnings en la actualización (esperado)${NC}"
fi
echo ""

# Step 3: Install PyTorch with CUDA 11.8 first
echo -e "${BLUE}[3/5] Instalando PyTorch 2.4.1+cu118 (esto toma varios minutos)...${NC}"
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 --quiet
if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ PyTorch instalado: $TORCH_VERSION${NC}"
    CUDA_AVAILABLE=$(python -c "import torch; print('Sí' if torch.cuda.is_available() else 'No')")
    echo -e "${GREEN}✓ CUDA disponible: $CUDA_AVAILABLE${NC}"
else
    echo -e "${RED}✗ Error instalando PyTorch${NC}"
    exit 1
fi
echo ""

# Step 4: Install remaining packages from freeze
echo -e "${BLUE}[4/5] Instalando paquetes desde freeze (129 paquetes)...${NC}"
if [ ! -f "$SCRIPT_DIR/dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt" ]; then
    echo -e "${RED}✗ Archivo de freeze no encontrado: $SCRIPT_DIR/dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt${NC}"
    exit 1
fi

pip install -r "$SCRIPT_DIR/dgarcia_r2gengpt_llama3.2_1b_pip_freeze.txt" --quiet
if [ $? -eq 0 ]; then
    PACKAGE_COUNT=$(pip list | wc -l)
    echo -e "${GREEN}✓ Todos los paquetes instalados ($PACKAGE_COUNT)${NC}"
else
    echo -e "${YELLOW}⚠ Algunos paquetes tuvieron problemas (algunos son opcionales)${NC}"
fi
echo ""

# Step 5: Verify installation
echo -e "${BLUE}[5/5] Verificando instalación...${NC}"
python << 'EOF'
import sys

checks = []

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ PyTorch: {e}")
    checks.append(False)

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ Transformers: {e}")
    checks.append(False)

try:
    import lightning
    print(f"✓ Lightning: {lightning.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ Lightning: {e}")
    checks.append(False)

try:
    import bitsandbytes
    print(f"✓ Bitsandbytes: {bitsandbytes.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ Bitsandbytes: {e}")
    checks.append(False)

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
    checks.append(True)
except Exception as e:
    print(f"✗ PEFT: {e}")
    checks.append(False)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA disponible: {'Sí' if cuda_available else 'No'}")
    checks.append(True)
except Exception as e:
    print(f"✗ CUDA check: {e}")
    checks.append(False)

if all(checks):
    print("\n✅ Verificación exitosa - Entorno listo para usar")
    sys.exit(0)
else:
    print("\n⚠ Algunas verificaciones fallaron (algunos paquetes son opcionales)")
    sys.exit(1)
EOF

VERIFY_RESULT=$?
echo ""

# Final summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    RESUMEN FINAL                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Entorno creado en: $VENV_PATH${NC}"
echo -e "${GREEN}✓ Python: $(python --version)${NC}"
echo -e "${GREEN}✓ Paquetes: $(pip list | wc -l) instalados${NC}"
echo ""

if [ $VERIFY_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ ENTORNO LISTO PARA USAR${NC}"
else
    echo -e "${YELLOW}⚠ Verificación completada con algunos warnings${NC}"
fi

echo ""
echo "Para activar el entorno en futuras sesiones:"
echo -e "${YELLOW}  source $VENV_PATH/bin/activate${NC}"
echo ""

# Keep venv activated
echo -e "${BLUE}El entorno está activado en esta sesión.${NC}"
echo "Deactivate con: ${YELLOW}deactivate${NC}"
echo ""
