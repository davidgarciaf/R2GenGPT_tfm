# Análisis de Conflictos de Dependencias - Llama2-7B Cuantizado

## 📊 Matriz de Compatibilidad

### Versiones Seleccionadas

| Componente | Versión | Razón de la Selección | Compatibilidad |
|-----------|---------|----------------------|-----------------|
| **Python** | 3.8+ | Soporte mínimo para torch 2.0+ | ✓ OK |
| **PyTorch** | 2.4.1 + CUDA 11.8 | Requerido para bitsandbytes (cuantización) | ✓ OK |
| **transformers** | 4.30.2 | Estable con PEFT y compatible con Llama2 | ✓ OK |
| **peft** | ≥0.4.0 | LoRA support, compatible con transformers 4.30.2 | ✓ OK |
| **bitsandbytes** | ≥0.40.0 | Cuantización 4-bit/8-bit (requiere CUDA 11.8+) | ✓ OK |
| **accelerate** | ≥0.20.0 | Distributed training, compatible con PyTorch 2.0+ | ✓ OK |
| **lightning** | 2.0.5 | Compatible con PyTorch 2.0+, stable API | ✓ OK |

---

## 🔴 Conflictos Potenciales y Cómo Evitarlos

### 1. ⚠️ PyTorch ↔ bitsandbytes (CRÍTICO)

**Problema**: bitsandbytes requiere CUDA 11.8 o superior
**Solución**: Instalar PyTorch con `--index-url https://download.pytorch.org/whl/cu118`

```bash
# ❌ INCORRECTO - Instalará PyTorch sin CUDA
pip install torch

# ✓ CORRECTO - Instalará PyTorch con CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Síntoma si no se hace correctamente**:
```
RuntimeError: CUDA out of memory / bitsandbytes not available
```

---

### 2. ⚠️ transformers ↔ peft

**Problema**: Versiones incompatibles de transformers pueden romper PEFT
**Solución**: Usar transformers==4.30.2 (versión conocida como estable)

```bash
# ❌ EVITAR - Puede tener incompatibilidades
pip install transformers  # Instalaría la última (5.x)

# ✓ CORRECTO - Versión pinned probada
pip install transformers==4.30.2
```

**Síntoma si no se hace correctamente**:
```
ImportError: cannot import name 'PrefixTuningConfig' from peft
```

---

### 3. ⚠️ PyTorch ↔ Lightning

**Problema**: PyTorch Lightning 2.0.5 requiere PyTorch ≥1.13
**Solución**: Automático si instalas en orden correcto

```bash
# Orden correcto:
pip install torch --index-url https://download.pytorch.org/whl/cu118  # 2.4.1
pip install lightning==2.0.5  # Automáticamente compatible
```

---

### 4. ⚠️ NumPy ↔ PyTorch

**Problema**: Cambios en NumPy 2.0 pueden afectar PyTorch
**Solución**: Dejar que pip resuelva automáticamente (compatible)

```bash
pip install numpy  # pip instalará compatible automáticamente
```

---

## 🟡 Conflictos de Grano Fino

### transformers 4.30.2 requiere:
- `tokenizers>=0.13.3` ✓ (automático)
- `safetensors>=0.3.1` ✓ (automático)
- `huggingface-hub<1.0,>=0.16.4` ✓ (automático)

### peft ≥0.4.0 requiere:
- `transformers` (cualquier versión compatible) ✓
- `torch>=1.13.0` ✓ (tenemos 2.4.1)
- `numpy` ✓

### bitsandbytes ≥0.40.0 requiere:
- CUDA 11.8+ (proporcionado por PyTorch) ✓
- `torch>=1.9.0` ✓
- `numpy` ✓

---

## ✅ Orden Correcto de Instalación

**IMPORTANTE**: El orden de instalación es crítico para resolver dependencias correctamente.

```bash
# 1. Actualizar herramientas base (PRIMERO)
pip install --upgrade pip setuptools wheel

# 2. PyTorch con CUDA (SEGUNDO - CRÍTICO)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Transformers con versión específica (TERCERO)
pip install transformers==4.30.2

# 4. PEFT para LoRA (CUARTO)
pip install peft>=0.4.0

# 5. Bitsandbytes para cuantización (QUINTO)
pip install bitsandbytes>=0.40.0

# 6. Resto de dependencias (SEXTO)
pip install peft>=0.4.0 lightning==2.0.5 tensorboardX gradio numpy Pillow accelerate
```

---

## 🧪 Verificación de Compatibilidad

Después de instalar, ejecuta:

```bash
# Verificar PyTorch y CUDA
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CuDNN version: {torch.backends.cudnn.version()}")
EOF

# Verificar transformers y bitsandbytes
python3 << 'EOF'
import transformers
import bitsandbytes
import peft
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print("bitsandbytes: OK")
EOF

# Verificar PyTorch Lightning
python3 << 'EOF'
import lightning
print(f"Lightning: {lightning.__version__}")
EOF
```

---

## 🔧 Resolución de Conflictos Existentes

Si ya instalaste algo y hay conflictos:

```bash
# Limpiar caché de pip
pip cache purge

# Desinstalar todo
pip uninstall torch transformers peft bitsandbytes accelerate lightning tensorboardX -y

# Reinstalar en orden correcto
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate
bash /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/install_complete.sh
```

---

## 📋 Dependencias Transitivas Completas

```
torch==2.4.1+cu118
├── filelock
├── typing-extensions>=4.8.0
├── sympy
├── networkx
├── jinja2
├── fsspec
├── nvidia-cuda-nvrtc-cu11==11.8.89
├── nvidia-cuda-runtime-cu11==11.8.89
├── nvidia-cuda-cupti-cu11==11.8.87
├── nvidia-cudnn-cu11==9.1.0.70
└── nvidia-cublas-cu11==11.11.3.6

transformers==4.30.2
├── tokenizers>=0.13.3
├── safetensors>=0.3.1
├── huggingface-hub<1.0,>=0.16.4
├── numpy
├── tqdm
└── requests

peft>=0.4.0
├── torch>=1.13.0
└── numpy

bitsandbytes>=0.40.0
├── torch>=1.9.0
└── numpy

lightning==2.0.5
├── torch>=1.12.0
├── numpy
├── tensorboard>=2.9.1
└── pytorch-lightning

accelerate>=0.20.0
├── numpy
├── torch>=1.10.0
├── pyyaml
└── psutil
```

---

## 🎯 Notas Finales

1. **NO mezclar índices**: Usa SIEMPRE `--index-url https://download.pytorch.org/whl/cu118` para PyTorch
2. **NO actualizar transformers automáticamente**: Mantener 4.30.2 fijo
3. **Reiniciar kernel**: Después de instalar CUDA, reinicia Python
4. **Verificar CUDA**: Asegurate de que `torch.cuda.is_available()` devuelve `True`

---

**Última actualización**: 2026-02-19
