#!/bin/bash

# ============================================================================
# LLAMA2-7B CUANTIZADO - REFERENCIA RÁPIDA DE COMANDOS
# ============================================================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════╗
║    🦙 LLAMA2-7B CUANTIZADO - REFERENCIA RÁPIDA DE COMANDOS          ║
╚══════════════════════════════════════════════════════════════════════╝

ENTORNO
═══════════════════════════════════════════════════════════════════════

Activar entorno:
  source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

Desactivar entorno:
  deactivate

Ver información del entorno:
  python3 -c "import sys; print(sys.prefix)"

Listar paquetes instalados:
  pip list


VERIFICACIÓN
═══════════════════════════════════════════════════════════════════════

Test básico (sin descargar modelo):
  python3 test_llama2_quantized.py

Test completo (descarga y prueba el modelo):
  python3 test_llama2_quantized.py --full

Verificación rápida:
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
  python3 -c "import bitsandbytes; print('bitsandbytes: OK')"


DEMOSTRACIÓN
═══════════════════════════════════════════════════════════════════════

Demo rápida:
  python3 quick_start.py

Ver ejemplos de código:
  cat example_usage.py

Ejecutar ejemplos específicos:
  python3 -c "from example_usage import example_1_basic_load; model, tok = example_1_basic_load()"


CONFIGURACIÓN HUGGINGFACE
═══════════════════════════════════════════════════════════════════════

Login (OBLIGATORIO para descargar modelo):
  huggingface-cli login

Logout:
  huggingface-cli logout

Ver información de cuenta:
  huggingface-cli whoami

Descargar modelo manualmente:
  huggingface-cli download meta-llama/Llama-2-7b


CONFIGURACIÓN DE GPU Y MEMORIA
═══════════════════════════════════════════════════════════════════════

Ver estado de GPU:
  nvidia-smi

Ver memoria CUDA en Python:
  python3 -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB')"

Seleccionar GPU específica:
  export CUDA_VISIBLE_DEVICES=0

Optimizar memoria (si hay OOM):
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3000

Ver todas las variables de entorno:
  env | grep -E 'CUDA|PYTORCH'


DEVELOPMENT
═══════════════════════════════════════════════════════════════════════

Instalar dependencia adicional:
  pip install nombre-del-paquete

Reinstalar paquete específico:
  pip install --force-reinstall nombre-del-paquete

Limpiar caché de pip:
  pip cache purge

Ver dependencias de un paquete:
  pip show nombre-del-paquete

Generar requirements actualizado:
  pip freeze > requirements.txt


DESARROLLO DE CÓDIGO
═══════════════════════════════════════════════════════════════════════

Lanzar Python interactivo:
  python3

Ejecutar script:
  python3 mi_script.py

Ejecutar con profiler:
  python3 -m cProfile -s cumtime mi_script.py

Debug con pdb:
  python3 -m pdb mi_script.py


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

Si hay error de CUDA:
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:5000
  python3 tu_script.py

Si hay error de modelo no encontrado:
  huggingface-cli login
  huggingface-cli download meta-llama/Llama-2-7b

Si hay conflicto de dependencias:
  pip cache purge
  pip install --force-reinstall torch transformers bitsandbytes

Reinstalar entorno completo:
  rm -rf venv/
  python3 -m venv venv
  source venv/bin/activate
  bash install_complete.sh


INFORMACIÓN Y DOCUMENTACIÓN
═══════════════════════════════════════════════════════════════════════

Ver README:
  cat README.md

Ver guía de instalación:
  cat INSTALL_GUIDE.md

Ver análisis de conflictos:
  cat CONFLICT_RESOLUTION.md

Ver reporte de verificación:
  cat VERIFICATION_REPORT.md

Ver índice de archivos:
  cat INDEX.md


TAMAÑOS Y ESTADÍSTICAS
═══════════════════════════════════════════════════════════════════════

Tamaño del entorno:
  du -sh venv/

Tamaño total:
  du -sh .

Lista detallada de archivos:
  ls -lh *.py *.md *.sh

Mostrar archivos por tamaño:
  ls -lhS


VARIABLES DE ENTORNO ÚTILES
═══════════════════════════════════════════════════════════════════════

# En ~/.bashrc o al inicio de script:

# GPU a usar
export CUDA_VISIBLE_DEVICES=0

# Configurar memoria CUDA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3000

# Desabilitar OpenMP si hay conflictos
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Usar float16 por defecto
export TF_CPP_MIN_LOG_LEVEL=3


ESTRUCTURAS DE CARPETAS
═══════════════════════════════════════════════════════════════════════

Ubicación del entorno:
  /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/

Carpeta del proyecto principal:
  /mnt/sd5/users/dgarcia/R2GenGPT/

Archivos de script:
  scripts/

Archivos de datos:
  dataset/

Modelos guardados:
  models/

Logs de entrenamiento:
  logs/


ACCESOS RÁPIDOS
═══════════════════════════════════════════════════════════════════════

Ir a la carpeta del entorno:
  cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

Activar con un comando:
  source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

Alias útil (agregar a ~/.bashrc):
  alias activate_llama2='source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate'
  alias llama2_test='python3 /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/test_llama2_quantized.py'
  alias llama2_quick='python3 /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/quick_start.py'


FLUJO DE TRABAJO TÍPICO
═══════════════════════════════════════════════════════════════════════

1. Activar entorno:
   source venv/bin/activate

2. Verificar GPU:
   nvidia-smi

3. Configurar HuggingFace (primera vez):
   huggingface-cli login

4. Ejecutar test:
   python3 test_llama2_quantized.py

5. Ejecutar tu código:
   python3 mi_script.py

6. Deactivar:
   deactivate


═══════════════════════════════════════════════════════════════════════
Última actualización: 2026-02-19
Estado: ✅ Completado y verificado
═══════════════════════════════════════════════════════════════════════

EOF
