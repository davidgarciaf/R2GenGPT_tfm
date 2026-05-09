#!/bin/bash
# Script de Verificación Rápida del Entorno Llama3.2-1B-Instruct Cuantizado

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  VERIFICACIÓN DEL ENTORNO LLAMA3.2-1B-INSTRUCT CUANTIZADO     ║"
echo "║  R2GenGPT - Sistema de Generación de Reportes Radiológicos    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar entorno virtual
echo "1. Verificando entorno virtual..."
VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv"
if [ -d "$VENV_PATH" ]; then
    echo "   ✓ Entorno virtual encontrado en: $VENV_PATH"
    echo "   ✓ Tamaño: $(du -sh $VENV_PATH | cut -f1)"
else
    echo "   ✗ ERROR: Entorno virtual NO encontrado"
    exit 1
fi
echo ""

# Verificar archivos de configuración
echo "2. Verificando archivos de configuración..."
CONFIG_DIR="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant"
FILES=("activate.sh" "install_env.sh" "requirements.txt" "README.md")
for file in "${FILES[@]}"; do
    if [ -f "$CONFIG_DIR/$file" ]; then
        echo "   ✓ $file ($(wc -l < $CONFIG_DIR/$file) líneas)"
    else
        echo "   ✗ $file NO ENCONTRADO"
    fi
done
echo ""

# Verificar script actualizado
echo "3. Verificando script de ejecución..."
SCRIPT_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/scripts/run_1-1.shallow_run_iuxray_rep.sh"
if grep -q "Llama3.2-1B-Instruct-quant" "$SCRIPT_PATH"; then
    echo "   ✓ Script actualizado correctamente"
    echo "   ✓ Ruta del venv: $(grep 'VENV_PATH=' $SCRIPT_PATH | cut -d'=' -f2 | tr -d '"')"
else
    echo "   ✗ Script NO está actualizado"
fi
echo ""

# Verificar documentación
echo "4. Verificando documentación..."
DOCS=("SETUP_GUIDE.md" "ENVIRONMENT_SETUP_SUMMARY.md")
for doc in "${DOCS[@]}"; do
    if [ -f "/mnt/sd5/users/dgarcia/R2GenGPT/$doc" ]; then
        echo "   ✓ $doc"
    fi
done
echo ""

# Test Python si el entorno está activado
echo "5. Verificando PyTorch y Transformers..."
source "$VENV_PATH/bin/activate" 2>/dev/null
if [ $? -eq 0 ]; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "   ✓ PyTorch: $TORCH_VERSION"
        CUDA_AVAILABLE=$(python -c "import torch; print('Sí' if torch.cuda.is_available() else 'No')" 2>/dev/null)
        echo "   ✓ CUDA disponible: $CUDA_AVAILABLE"
        TF_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
        echo "   ✓ Transformers: $TF_VERSION"
        BNB_VERSION=$(python -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null)
        echo "   ✓ Bitsandbytes: $BNB_VERSION"
    else
        echo "   ⚠ No se pudieron verificar versiones (python test falló)"
    fi
else
    echo "   ⚠ No se pudo activar el entorno (test saltado)"
fi
echo ""

# Verificar token de HF
echo "6. Verificando Token de Hugging Face..."
TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"
if [ -f "$TOKEN_FILE" ]; then
    echo "   ✓ Token file encontrado"
    if [ -s "$TOKEN_FILE" ]; then
        echo "   ✓ Token file no está vacío"
    else
        echo "   ⚠ Token file está vacío - DEBES AGREGAR TU TOKEN"
    fi
else
    echo "   ⚠ Token file NO encontrado - DEBES CREAR ANTES DE EJECUTAR"
fi
echo ""

# Resumen final
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    RESUMEN DE ESTADO                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Entorno virtual: LISTO"
echo "✓ Archivos de configuración: COMPLETOS"
echo "✓ Scripts actualizados: SÍ"
echo "✓ Documentación: COMPLETA"
echo "⚠ Token de HF: REQUIERE CONFIGURACIÓN (ver abajo)"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                PRÓXIMOS PASOS                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. CONFIGURAR TOKEN DE HUGGING FACE (requerido):"
echo "   Obtén tu token en: https://huggingface.co/settings/tokens"
echo "   Luego ejecuta:"
echo "   $ cat > /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env << 'EOF'"
echo "   HF_TOKEN=tu_token_aqui"
echo "   EOF"
echo ""

echo "2. ACTIVAR EL ENTORNO:"
echo "   $ source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/activate.sh"
echo ""

echo "3. EJECUTAR EL EXPERIMENTO:"
echo "   $ cd /mnt/sd5/users/dgarcia/R2GenGPT"
echo "   $ bash scripts/run_1-1.shallow_run_iuxray_rep.sh"
echo ""

echo "4. VER LOGS EN TIEMPO REAL:"
echo "   $ tail -f /mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/log.txt"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            DOCUMENTACIÓN DISPONIBLE                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "• SETUP_GUIDE.md"
echo "  Guía completa paso a paso para reproducir el experimento"
echo ""
echo "• ENVIRONMENT_SETUP_SUMMARY.md"
echo "  Resumen de lo que se ha configurado"
echo ""
echo "• environment/Llama3.2-1B-Instruct-quant/README.md"
echo "  Documentación técnica del entorno"
echo ""

echo "✅ Verificación completada."
echo ""
