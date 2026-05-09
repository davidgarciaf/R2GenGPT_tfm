#!/bin/bash
# Script con reinicio completo de CUDA para resolver problemas de estado corrupto

# Limpiar completamente el entorno CUDA
unset CUDA_VISIBLE_DEVICES
unset CUDA_LAUNCH_BLOCKING
unset CUDA_CACHE_DISABLE
unset CUDA_CACHE_PATH
unset PYTORCH_CUDA_ALLOC_CONF
unset TORCH_USE_CUDA_DSA

# Crear un entorno CUDA completamente limpio
export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_DISABLE=1
export CUDA_CACHE_PATH=/tmp/cuda_cache_clean_$(date +%s)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1

# Limpiar cualquier caché CUDA existente
echo "Limpiando caché CUDA existente..."
rm -rf /tmp/cuda_cache_*
rm -rf ~/.cache/torch/kernels
rm -rf ~/.nv/ComputeCache

# Crear directorio de caché limpio
mkdir -p $CUDA_CACHE_PATH

# Activar entorno Virtual con las dependencias cuantizadas
echo "Activando entorno virtual para Llama3.2-1B-Instruct cuantizado..."
VENV_PATH="/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: No se encontró el entorno virtual en $VENV_PATH"
    echo "Por favor, ejecuta: bash /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/install_env.sh"
    exit 1
fi
source "$VENV_PATH/bin/activate"

# Forzar recarga completa de módulos CUDA
echo "Forzando recarga de módulos CUDA..."
python -c "
import sys
import importlib
modules_to_reload = ['torch', 'torchvision', 'torchaudio']
for mod in modules_to_reload:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
print('Módulos recargados')
" 2>/dev/null || true

# Verificar CUDA con diagnóstico detallado
echo "=== DIAGNÓSTICO CUDA ==="
python -c "
import os
print('Variables de entorno CUDA:')
for k, v in os.environ.items():
    if 'CUDA' in k or 'TORCH' in k:
        print(f'  {k}={v}')

print('\nVerificando PyTorch...')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Número de GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA NO DISPONIBLE')
" || {
    echo "ERROR: Falló la verificación de CUDA"
    exit 1
}

# Prueba intensiva de CUDA
echo "=== PRUEBA INTENSIVA DE CUDA ==="
python -c "
import torch
import gc

print('Limpiando memoria GPU...')
gc.collect()
torch.cuda.empty_cache()

print('Creando tensores de prueba...')
try:
    # Prueba básica
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print(f'✓ Multiplicación básica: {z.shape}')
    
    # Prueba de memoria
    big_tensor = torch.randn(1000, 1000).cuda()
    print(f'✓ Tensor grande creado: {big_tensor.shape}')
    
    # Limpiar
    del x, y, z, big_tensor
    gc.collect()
    torch.cuda.empty_cache()
    print('✓ Memoria limpiada correctamente')
    
except Exception as e:
    print(f'✗ Error en prueba CUDA: {e}')
    exit(1)

print('✓ Todas las pruebas de CUDA pasaron')
" || {
    echo "ERROR: Fallaron las pruebas de CUDA"
    exit 1
}

echo "=== CUDA VERIFICADO CORRECTAMENTE ==="

# Ir al directorio del proyecto
cd /mnt/sd5/users/dgarcia/R2GenGPT

# Cargar el token de Hugging Face del archivo privado
TOKEN_FILE="/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env"

# Cargar el token de Hugging Face del archivo privado
if [ -f "$TOKEN_FILE" ]; then
    export $(cat $TOKEN_FILE | xargs)
    echo "Token exportado desde el archivo $TOKEN_FILE"
else
    echo "ERROR: No se encontró el archivo $TOKEN_FILE con el token de Hugging Face."
    exit 1
fi

dataset="iu_xray"
annotation="/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json"
base_dir="/mnt/sd5/users/dgarcia/data/iu_xray/images"

version="v1_shallow_rep" # Directorio de salida modificado para evitar sobrescritura
savepath="/mnt/sd5/users/dgarcia/R2GenGPT/save/$dataset/$version"

# Ensure save directory exists before tee attempts to write
mkdir -p "$savepath"

# ------------------------------------------------------------------
# Control de modalidad: quick test vs entrenamiento completo
# Para una prueba rápida en la cola, exporta QUICK=1 antes de llamar
# al script; de lo contrario se hará la ejecución completa (shallow).
# ------------------------------------------------------------------
if [ "${QUICK}" = "1" ]; then
    bs=2
    vbs=2
    devs=1
    strat="auto"
    maxep=3
    limval=0.0
    workers=4
else
    bs=4
    vbs=4
    devs=1
    strat="auto"
    maxep=15
    limval=1.0
    workers=2
fi


# -------------------------------------------------------------

# -------------------------------------------------------------
# Añadimos ajustes para utilizar el modelo cuantizado (4-bit) y
# que este script pueda ejecutarse en la cola del clúster.
# Se puede enviar con:
#   qsub -q tfm.q@deimos run_1-1.shallow_run_iuxray_rep.sh
# Asegúrate de tener una línea #$ con la cola y memoria si lo usas.
# -------------------------------------------------------------

# Parámetros de cuantización que se pasarán a train.py
# Nota: double_quant=False reduce consumo de memoria temporal durante carga
quant_opts="--load_in_4bit True \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_use_double_quant False \
    --bnb_4bit_quant_type nf4"

python -u /mnt/sd5/users/dgarcia/R2GenGPT/train.py \
    --llama_model "meta-llama/Llama-3.2-1B-Instruct" \
    --low_resource True \
    --precision 16-mixed \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size ${bs} \
    --val_batch_size ${vbs} \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers ${workers} \
    --devices ${devs} \
    --strategy ${strat} \
    --max_epochs ${maxep} \
    --limit_val_batches ${limval} \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 0 \
    ${quant_opts} \
    2>&1 |tee -a ${savepath}/log.txt
