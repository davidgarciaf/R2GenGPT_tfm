# Guía de Reproducción de Experimento R2GenGPT con Llama3.2-1B-Instruct Cuantizado

## Resumen de lo que se ha hecho

Se ha creado un nuevo entorno virtualizado optimizado bajo:
```
/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant
```

Este entorno contiene:
- **PyTorch 2.4.1+cu118** con soporte CUDA 11.8
- **Transformers 4.30.2** para cargar modelos de HuggingFace
- **Lightning 2.0.5** para entrenamiento distribuido
- **Bitsandbytes 0.45.5** para cuantización 4-bit NF4
- **PEFT 0.13.2** para fine-tuning eficiente con LoRA
- Todas las dependencias necesarias para R2GenGPT

El script de ejecución ha sido actualizado:
```
/mnt/sd5/users/dgarcia/R2GenGPT/scripts/run_1-1.shallow_run_iuxray_rep.sh
```

## Paso 1: Configurar Token de Hugging Face

El modelo Llama3.2-1B-Instruct requiere autenticación. Necesitas crear un archivo con tu token:

```bash
# 1. Obtén tu token en: https://huggingface.co/settings/tokens
# 2. Reemplaza "tu_token_aqui" con tu token real
# 3. Ejecuta esto en la terminal:

cat > /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env << 'EOF'
HF_TOKEN=tu_token_aqui
EOF

# 4. Verifica que el archivo se creó correctamente:
cat /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env
```

## Paso 2: Activar el Entorno

Abre una terminal nueva y ejecuta:

```bash
# Opción A: Usando el script helper (recomendado - limpia estado CUDA)
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/activate.sh

# Opción B: Activación manual
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv/bin/activate
```

Deberías ver `(venv)` al inicio de tu línea de comando.

## Paso 3: Verificar la Configuración del Entorno

Ejecuta este test para confirmar que todo está bien:

```bash
python << 'EOF'
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("VERIFICACIÓN DEL ENTORNO LLAMA3.2-1B-INSTRUCT CUANTIZADO")
print("=" * 60)

# 1. PyTorch
print("\n✓ PyTorch:")
print(f"  - Versión: {torch.__version__}")
print(f"  - CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - GPUs detectadas: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"    * GPU {i}: {props.name}")
        print(f"      Memoria: {props.total_memory / 1024**3:.1f} GB")

# 2. Transformers
from transformers import __version__ as transformers_version
print(f"\n✓ Transformers: {transformers_version}")

# 3. Cuantización 4-bit
print("\n✓ Configuración de Cuantización 4-bit:")
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"
)
print(f"  - load_in_4bit: {config.load_in_4bit}")
print(f"  - bnb_4bit_quant_type: {config.bnb_4bit_quant_type}")
print(f"  - bnb_4bit_compute_dtype: {config.bnb_4bit_compute_dtype}")

# 4. Tokenizer
print("\n✓ Descargando tokenizer (primera vez)...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    trust_remote_code=True
)
print(f"  - Tokenizer OK: vocab_size={tokenizer.vocab_size}")

print("\n" + "=" * 60)
print("✅ ENTORNO COMPLETAMENTE CONFIGURADO Y LISTO")
print("=" * 60)
EOF
```

**Salida esperada**: Deberías ver checkmarks verdes (✓) indicando que todo está OK.

## Paso 4: Ejecutar el Experimento (Shallow - 15 épocas)

### Opción A: Ejecución Directa (Recomendado para pruebas)

```bash
cd /mnt/sd5/users/dgarcia/R2GenGPT
bash scripts/run_1-1.shallow_run_iuxray_rep.sh
```

El entrenamiento comenzará en aproximadamente 5-10 minutos (tiempo para cargar modelos).

### Opción B: Quick Test (3 épocas)

```bash
cd /mnt/sd5/users/dgarcia/R2GenGPT
QUICK=1 bash scripts/run_1-1.shallow_run_iuxray_rep.sh
```

Útil para verificar que todo funciona antes de ejecutar el entrenamiento completo.

### Opción C: Enviar a la Cola del Clúster

```bash
qsub -q tfm.q@deimos /mnt/sd5/users/dgarcia/R2GenGPT/scripts/run_1-1.shallow_run_iuxray_rep.sh
```

## Monitoreo del Entrenamiento

Los logs se guardan en tiempo real en:
```
/mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/log.txt
```

Para ver el progreso en vivo:
```bash
tail -f /mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/log.txt
```

O con timestamp:
```bash
tail -f /mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/log.txt | grep -E "Epoch|loss|val"
```

## Parámetros del Experimento

El script usa estos parámetros (shallow training):

| Parámetro | Valor |
|-----------|-------|
| **Modelo LLM** | meta-llama/Llama-3.2-1B-Instruct |
| **Cuantización** | 4-bit NF4 |
| **Batch Size** | 4 (training) / 4 (validation) |
| **Máximo de épocas** | 15 |
| **Max tokens generados** | 100 |
| **Min tokens generados** | 40 |
| **Precisión** | 16-bit mixed |
| **Estrategia** | auto (ddp/single GPU) |
| **Freeze Vision Encoder** | True |
| **Vision Model** | microsoft/swin-base-patch4-window7-224 |

Para cambiar parámetros, edita el archivo:
```bash
nano /mnt/sd5/users/dgarcia/R2GenGPT/scripts/run_1-1.shallow_run_iuxray_rep.sh
```

Secciones importantes:
- Línea 55-65: Batch sizes y número de épocas
- Línea 78-89: Parámetros de cuantización
- Línea 90-107: Parámetros del modelo y entrenamiento

## Descripción de Directorios Clave

```
/mnt/sd5/users/dgarcia/R2GenGPT/
├── environment/Llama3.2-1B-Instruct-quant/
│   ├── venv/                  # Entorno virtual con todas las dependencias
│   ├── activate.sh            # Script para activar el entorno
│   ├── install_env.sh         # Script para reinstalar desde cero
│   ├── requirements.txt       # Lista de paquetes Python
│   └── README.md              # Documentación técnica completa
│
├── save/iu_xray/
│   ├── v1_shallow/            # Experimento anterior (referencia)
│   └── v1_shallow_rep/        # NUEVO: Experimento de reproducción
│       ├── checkpoints/       # Checkpoints de entrenamiento
│       ├── logs/              # Logs de Lightning
│       ├── result/            # Resultados finales
│       └── log.txt            # Log completo del entrenamiento
│
└── scripts/
    ├── run_1-1.shallow_run_iuxray_rep.sh  # ACTUALIZADO para nuevo entorno
    ├── run_1-2.shallow_test_iuxray.sh     # Testing script
    └── ... otros scripts
```

## Solución de Problemas

### Problema 1: Token de Hugging Face no encontrado

```bash
ERROR: No se encontró el archivo /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env
```

**Solución**:
```bash
# Crea el archivo con tu token
cat > /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama3.env << 'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
EOF
```

### Problema 2: CUDA out of memory

```bash
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Soluciones**:

Opción A: Reducir batch size (Edita run_1-1.shallow_run_iuxray_rep.sh)
```bash
bs=2      # Cambiar de 4 a 2
vbs=2     # Cambiar de 4 a 2
```

Opción B: Ejecutar quick test
```bash
QUICK=1 bash scripts/run_1-1.shallow_run_iuxray_rep.sh
```

Opción C: Limpiar memoria CUDA
```bash
# Salir del terminal y activar el entorno de nuevo
exit
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/activate.sh
```

### Problema 3: "No module named 'transformers'"

**Solución**:
```bash
# Verificar activación del entorno
which python  # Debe mostrar ruta dentro de venv
pip list | grep transformers  # Debe mostrar transformers 4.30.2
```

Si no aparece, reinstala:
```bash
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/venv/bin/activate
pip install transformers==4.30.2 --force-reinstall
```

### Problema 4: Errores de compilación CUDA

```bash
RuntimeError: CUDA kernel compilation failed
```

**Solución**: Limpiar caché CUDA
```bash
# Usar el script de activación que limpia estado CUDA
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama3.2-1B-Instruct-quant/activate.sh
```

## Comparación con Experimento Anterior

| Aspecto | v1_shallow | v1_shallow_rep (Nueva) |
|--------|-----------|----------------------|
| **Entorno** | Conda dgarcia_tfm_clean | venv Llama3.2-1B-Instruct-quant |
| **PyTorch** | Anterior | 2.4.1+cu118 |
| **Transformers** | 4.52.4 | 4.30.2 |
| **Estado Inicial** | Corrupto/Problemas CUDA | Limpio y verificado |
| **Cuantización** | Sí (4-bit) | Sí (4-bit NF4) |
| **Versión Llama** | Llama3.2-1B-Instruct | Llama3.2-1B-Instruct |
| **Resultados** | Disponibles en logs/ | Se guardarán en v1_shallow_rep/ |

## Checkpoint de Progreso

El experiment completo (15 épocas con 2 GPUs) tomará aproximadamente:
- **Tiempo total**: 3-5 horas
- **Tiempo por época**: 12-20 minutos
- **Primeros resultados**: ~30 minutos de ejecución

Los checkpoints se guardan automáticamente cada época en:
```
/mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/checkpoints/
```

## Información Adicional

- **Documentación del Entorno**: Ver [environment/Llama3.2-1B-Instruct-quant/README.md](../environment/Llama3.2-1B-Instruct-quant/README.md)
- **Logs del Experimento Anterior**: `/mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow/logs/`
- **Código del Modelo**: `/mnt/sd5/users/dgarcia/R2GenGPT/models/R2GenGPT.py`
- **Configuración de Entrenamiento**: `/mnt/sd5/users/dgarcia/R2GenGPT/configs/config.py`

## Próximos Pasos

1. ✅ Crear entorno virtual cuantizado
2. ✅ Instalar todas las dependencias
3. ✅ Actualizar scripts de ejecución
4. ⏳ **AHORA**: Configurar token de HF y ejecutar el experimento

¡Estás listo para reproducir el experimento!
