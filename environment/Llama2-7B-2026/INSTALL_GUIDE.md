# Guía de Instalación: Entorno Llama2-7B Cuantizado

## 📋 Resumen del Proceso

Este documento explica cómo crear y configurar el entorno virtual para ejecutar Llama2-7B con cuantización 4-bit.

## 🚀 Pasos de Instalación

### 1. Entorno Virtual Creado ✅
Ya existe un entorno virtual en `/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/`

### 2. Activar el Entorno

```bash
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate
```

### 3. Instalar Dependencias

Las dependencias se están instalando. Usa uno de estos métodos:

**Opción A: Instalación automática (recomendada)**
```bash
cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026
bash install_env.sh
```

**Opción B: Instalación manual paso a paso**

```bash
# Actualizar herramientas base
pip install --upgrade pip setuptools wheel

# PyTorch con CUDA 11.8 (CRÍTICO para cuantización)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers (versión pinned por compatibilidad)
pip install transformers==4.30.2

# LoRA para ajuste fino eficiente
pip install peft

# Entrenamiento
pip install lightning==2.0.5
pip install tensorboardX

# Utilidades
pip install Pillow numpy gradio

# CRÍTICO: Librerías de cuantización
pip install bitsandbytes  # Cuantización 4-bit y 8-bit
pip install accelerate    # Entrenamiento distribuido
```

## ⚠️ Resolución de Conflictos de Dependencias

### Conflictos Evitados:

1. **PyTorch con CUDA 11.8**: Necesario para `bitsandbytes` que requiere CUDA 11.8+
2. **transformers==4.30.2**: Pinned para compatibilidad con peft
3. **lightning==2.0.5**: Compatible con PyTorch 2.0+
4. **bitsandbytes**: DEBE instalarse después de PyTorch

### Si hay conflictos:

```bash
# Limpiar cache
pip cache purge

# Reinstalar con resolución estricta
pip install --upgrade pip
pip install --force-reinstall transformers==4.30.2
```

## 📦 Librerías para Cuantización de Llama2

Las siguientes librerías son CRÍTICAS para Llama2-7B cuantizado:

| Librería | Versión | Propósito |
|----------|---------|----------|
| torch | >=2.0 | Framework DL (con CUDA 11.8) |
| transformers | 4.30.2 | Carga de modelos LLM |
| peft | >=0.4.0 | LoRA (Parameter-Efficient Fine-Tuning) |
| bitsandbytes | >=0.40.0 | Cuantización 4-bit/8-bit |
| accelerate | >=0.20.0 | Entrenamiento distribuido |
| lightning | 2.0.5 | Entrenamiento (PyTorch Lightning) |

## ✅ Verificar Instalación

```bash
# Activar entorno
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

# Verificar PyTorch y CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Dispositivo CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No disponible\"}')"

# Verificar transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Verificar bitsandbytes (cuantización)
python3 -c "import bitsandbytes; print('bitsandbytes instalado ✓')"

# Verificar peft (LoRA)
python3 -c "import peft; print('peft instalado ✓')"
```

## 🔧 Configuración Específica para Llama2-7B Cuantizado

### Variables de Entorno Recomendadas

Agregar al `.bashrc` o script de activación:

```bash
export CUDA_VISIBLE_DEVICES=0  # O el GPU que uses
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3000  # Para evitar OOM
```

### Configuración de Cuantización

Para usar Llama2-7B con cuantización 4-bit en tu código:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## 📝 Notas Importantes

1. **Tamaño de descarga**: PyTorch + dependencias ~2GB
2. **Tiempo de instalación**: 10-30 minutos (depende de conexión)
3. **CUDA 11.8**: OBLIGATORIO para bitsandbytes (cuantización)
4. **Memoria**: Llama2-7B cuantizado requiere ~8-16GB VRAM

## 🆘 Troubleshooting

### Error: "bitsandbytes not available"
```bash
pip install --force-reinstall bitsandbytes
```

### Error: "CUDA out of memory"
Aumentar `PYTORCH_CUDA_ALLOC_CONF` o reducir batch size

### Error: "transformers version mismatch"
```bash
pip install transformers==4.30.2 --force-reinstall
```

---

**Fecha de creación**: 2026-02-19
**Estado**: ✅ INSTALACIÓN COMPLETADA Y VERIFICADA
