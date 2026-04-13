# 🔧 SOLUCIONES DE OPTIMIZACIÓN PARA GTX 1080/1070

Si necesitas usar GTX 1080 o GTX 1070, aquí hay varias estrategias para que funcione.

## Problema

GTX 1080/1070 tienen Compute Capability 6.1, pero bitsandbytes recomienda 7.0+.

## Soluciones

### ✅ Solución 1: Usar load_in_8bit (Recomendada)

Modifica `submit_model_cluster.sh` línea 34:

```python
# CAMBIAR ESTO:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# A ESTO:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,  # 8-bit en lugar de 4-bit
)
```

**Ventajas:**
- ✅ Compatible con Maxwell (CC 6.1)
- ✅ Menos problemas de compilación
- ✅ Requiere ~9-10GB VRAM (ajustado pero funciona)

**Desventajas:**
- ❌ Inferencia más lenta que 4-bit
- ❌ Un poco más de memoria

---

### 🔧 Solución 2: Compilar bitsandbytes localmente

Si tienes problemas al importar bitsandbytes:

```bash
# En el nodo del cluster o máquina local
cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

source venv/bin/activate

# Desinstalar versión precompilada
pip uninstall -y bitsandbytes

# Instalar desde fuente
pip install bitsandbytes --no-binary :all: --force-reinstall

# Compilará con CPU kernels (más lento pero funciona)
```

**Ventajas:**
- ✅ Puede hacer que funcione bitsandbytes
- ✅ Sigue siendo 4-bit

**Desventajas:**
- ❌ Compilación toma 10-20 minutos
- ❌ CPU kernels = inferencia muy lenta

---

### 🚀 Solución 3: Usar versión modificada de bitsandbytes

Hay versiones compiladas específicamente para Maxwell:

```bash
# Desinstalar versión actual
pip uninstall -y bitsandbytes

# Instalar versión compilada para Maxwell
pip install bitsandbytes==0.39.0

# 0.39.0 tiene mejor soporte para GPUs más viejas
```

---

### 💡 Solución 4: Usar load_in_4bit sin double_quant

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,  # DESACTIVAR doble cuantización
    bnb_4bit_quant_type="nf4",
)
```

**Ventajas:**
- ✅ 4-bit sin problemas de compilación adicionales
- ✅ Faster that 8-bit

**Desventajas:**
- ❌ Menos compresión que con double_quant
- ❌ Un poco más VRAM (pero sigue siendo ~7-8GB)

---

## Recomendación

**Si usas GTX 1080/1070:**

1. **Primero intenta:** Solución 1 (load_in_8bit) - Más simple
2. **Si quieres 4-bit:** Solución 4 (sin double_quant)
3. **Si nada funciona:** Solución 2 (compilar desde source)

## Ejemplo Completo para GTX 1080

Archivo: `submit_model_gtx1080.sh`

```bash
#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/model_gtx1080.log
#$ -l h_vmem=10G

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

python3 << 'EOF'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

print("=" * 70)
print("  LLAMA2-7B EN GTX 1080 (8-bit)")
print("=" * 70)

# Cargar token
token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
with open(token_file, 'r') as f:
    for line in f:
        if line.startswith('HF_TOKEN='):
            token = line.replace('HF_TOKEN=', '').strip()
            os.environ['HF_TOKEN'] = token
            login(token=token, add_to_git_credential=False)

print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# CONFIGURACIÓN PARA GTX 1080: 8-bit
print("\n1. Configurando cuantización 8-bit para GTX 1080...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,  # ← 8-bit para mejor compatibilidad
)
print("   ✓ Configuración lista")

# Cargar tokenizer
print("\n2. Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    token=os.environ.get('HF_TOKEN')
)
print(f"   ✓ Tokenizer cargado (vocab: {len(tokenizer)})")

# Cargar modelo
print("\n3. Cargando modelo Llama2-7B con 8-bit...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("   ✓ Modelo cargado exitosamente")
    print(f"   - Parámetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Test de inferencia
    print("\n4. Test de inferencia...")
    prompt = "Machine learning is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✓ Inferencia completada")
    print(f"\n   Prompt: '{prompt}'")
    print(f"   Response: '{response}'")
    
    print("\n" + "=" * 70)
    print("✅ ÉXITO - Llama2-7B 8-bit funciona en GTX 1080")
    print("=" * 70)
    
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    print("\n⚠️  Si error es sobre bitsandbytes:")
    print("   pip install bitsandbytes==0.39.0")
    print("   O intenta compilar desde source:")
    print("   pip install bitsandbytes --no-binary :all: --force-reinstall")
EOF
```

**Usar:**
```bash
qsub -q student.q@pcgtx1080 submit_model_gtx1080.sh
```

---

## Monitorización en Cluster

```bash
# Ver jobs
qstat

# Ver output en vivo
tail -f logs/model_gtx1080.log

# Cancelar si es necesario
qdel <job_id>
```

---

## Si Nada Funciona

Si después de todo esto sigue sin funcionar en GTX 1080:

1. Intenta con GTX 1070 (por si acaso)
2. Intenta con un modelo más pequeño (Llama2-3B)
3. Solicita acceso a deimos/phobos
4. Usa Ollama o LLaMA.cpp (alternativas sin CUDA)

```bash
# Alternativa: LLaMA.cpp (CPU, lento pero siempre funciona)
pip install llama-cpp-python

# O Ollama (interfaz simplificada)
# https://ollama.ai
```

---

## Comparación de Estrategias

| Estrategia | Compatibilidad | Velocidad | VRAM | Complejidad |
|-----------|-----------------|-----------|------|-------------|
| load_in_8bit | ✅ Buena | 🟡 Lenta | ~9GB | ⭐ Simple |
| sin double_quant | ✅ Buena | 🟢 Media | ~8GB | ⭐ Simple |
| Compilar fuente | ✅ Funciona | 🔴 Muy lenta | ~7GB | ⭐⭐⭐ Complejo |
| bitsandbytes 0.39.0 | ✅ Buena | 🟢 Rápida | ~8GB | ⭐ Simple |

**Recomendación:** Usa `load_in_8bit` para máxima compatibilidad.

