# 📊 ANÁLISIS DE COMPATIBILIDAD - CLUSTER vs LLAMA2-7B

## Resumen Ejecutivo

**Recomendación Principal:** 🟢 **USAR COLA tfm.q (deimos o phobos)** si disponen de GPUs modernas (RTX series, V100, A100)

**Recomendación Secundaria:** 🟡 **USAR GTX 1080/1070** como fallback, con optimizaciones especiales

**Evitar:** 🔴 **GTX 970** - Insuficiente VRAM y Compute Capability antigua

---

## 📋 Especificaciones de Hardware

### 🔵 Cola: student.q

| Nodo | GPU | VRAM | CC* | CUDA | Recomendación |
|------|-----|------|-----|------|----------------|
| pcgtx1080 | GTX 1080 | 8GB | 6.1 | 11.1+ | 🟡 Marginal |
| pcgtx1070 | GTX 1070 | 8GB | 6.1 | 11.0+ | 🟡 Marginal |
| pcgtx970 | GTX 970 | 4GB | 5.2 | 10.2+ | 🔴 Incompatible |

*CC = Compute Capability (requerido ≥ 7.0 para bitsandbytes óptimo)

### 🔵 Cola: tfm.q

| Nodo | GPU | VRAM | CC | CUDA | Recomendación |
|------|-----|------|-----|------|----------------|
| deimos | ❓ Desconocido | ? | ? | ? | 🟢 Probablemente óptimo |
| phobos | ❓ Desconocido | ? | ? | ? | 🟢 Probablemente óptimo |

**Necesita verificación ejecutando:**
```bash
ssh deimos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv"
ssh phobos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv"
```

---

## 🔍 Análisis Detallado de Compatibilidad

### 1. GTX 1080 / GTX 1070 (student.q)

**Especificaciones:**
- VRAM: 8GB ✅
- Compute Capability: 6.1 ❌ (bitsandbytes requiere 7.0+)
- CUDA: 11.0+ ✅

**Compatibilidad con Llama2-7B 4-bit:**
- Memoria: ✅ Suficiente (necesita ~7-8GB)
- bitsandbytes: ⚠️ Problemas potenciales
  - Compute Capability 6.1 está por debajo del mínimo recomendado
  - Posibles errores de compilación al instalar bitsandbytes
  - Versiones antiguas podrían no compilarse correctamente

**Soluciones si usas GTX 1080/1070:**

**Opción A: Usar bitsandbytes en CPU (lento)**
```python
# En lugar de 4-bit cuantizado
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,  # Desactivar 4-bit
    load_in_8bit=False,  # Usar 16-bit
)
# Requiere ~14GB VRAM - NO VA A FUNCIONAR EN 8GB
```

**Opción B: Usar load_in_8bit (más optimizado)**
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,
)
# Requiere ~9-10GB - MARGINAL EN 8GB
```

**Opción C: Usar modelo más pequeño o pruning**
- Usar Llama2-3B en lugar de 7B
- Usar técnicas de pruning
- Usar knowledge distillation

**⚠️ Recomendación:** No es la opción ideal. Intenta primero con deimos/phobos.

---

### 2. GTX 970 (student.q@pcgtx970)

**Especificaciones:**
- VRAM: 4GB ❌
- Compute Capability: 5.2 ❌
- CUDA: 10.2+ ❌ (antiguo)

**Compatibilidad:**
- ❌ VRAM insuficiente (necesita ~7GB mínimo)
- ❌ Compute Capability demasiado antigua
- ❌ CUDA driver antiguo

**Conclusión:** 🔴 **NO USAR** - Completamente incompatible

---

### 3. Deimos y Phobos (tfm.q)

**Estado:** ❓ DESCONOCIDO - NECESITA VERIFICACIÓN

**Posibilidades probables:**
- GPU NVIDIA moderna (RTX series, V100, A100, H100)
- VRAM ≥ 12GB (típico en GPUs modernas)
- Compute Capability ≥ 7.0 (compatible con bitsandbytes)
- CUDA 11.8+ (soportado)

**Recomendación:** 🟢 **USAR ESTA COLA** si tienes acceso

**Para verificar exactamente:**
```bash
# Verificar información GPU
qstat -f -h deimos | grep -i gpu
qstat -f -h phobos | grep -i gpu

# O intentar conectar directamente
ssh deimos "nvidia-smi"
ssh phobos "nvidia-smi"

# Información detallada
ssh deimos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
ssh phobos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
```

---

## 📈 Comparativa de Rendimiento Esperado

| Nodo | GPU | Memoria | Compatibilidad | Velocidad Inferencia | Recommendation |
|------|-----|---------|-----------------|----------------------|----------------|
| deimos | ??? | ??? | 🟢 Probablemente sí | 🟢 Probablemente rápida | **1ª OPCIÓN** |
| phobos | ??? | ??? | 🟢 Probablemente sí | 🟢 Probablemente rápida | **2ª OPCIÓN** |
| pcgtx1080 | GTX 1080 | 8GB | 🟡 Con problemas | 🟡 Normal | 3ª OPCIÓN |
| pcgtx1070 | GTX 1070 | 8GB | 🟡 Con problemas | 🟡 Normal | 4ª OPCIÓN |
| pcgtx970 | GTX 970 | 4GB | 🔴 No | ❌ N/A | ❌ EVITAR |

---

## 🚀 Pasos Recomendados

### Paso 1: Verificar recursos en deimos/phobos
```bash
ssh deimos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
ssh phobos "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader"
```

### Paso 2: Enviar test inicial
```bash
# Si deimos/phobos tienen GPUs modernas:
qsub -q tfm.q@deimos submit_test_cluster.sh

# Si necesitas GTX 1080/1070:
qsub -q student.q@pcgtx1080 submit_test_cluster.sh
```

### Paso 3: Si el test pasa, enviar carga de modelo
```bash
qsub -q tfm.q@deimos submit_model_cluster.sh
```

### Paso 4: Si todo funciona, lanzar trabajo productivo
```bash
qsub -q tfm.q@deimos tu_script_principal.sh
```

---

## ⚙️ Scripts Disponibles

Se han creado los siguientes scripts para facilitar pruebas:

| Script | Propósito | Uso |
|--------|-----------|-----|
| check_cluster_resources.sh | Mostrar análisis de recursos | `bash check_cluster_resources.sh` |
| submit_test_cluster.sh | Test rápido de entorno | `qsub -q student.q@pcgtx1080 submit_test_cluster.sh` |
| submit_tokenizer_cluster.sh | Test de tokenizer (ligero) | `qsub -q tfm.q@deimos submit_tokenizer_cluster.sh` |
| submit_model_cluster.sh | Test de carga de modelo | `qsub -q tfm.q@deimos submit_model_cluster.sh` |

---

## 📝 Notas Importantes

### Sobre bitsandbytes y Compute Capability

bitsandbytes 0.41.0+ requiere Compute Capability ≥ 7.0 para:
- CUDA kernels optimizados
- Cuantización 4-bit/8-bit eficiente
- Soporte de mixed precision

GTX 1080/1070 (CC 6.1) pueden tener problemas:
- La compilación podría fallar
- O funcionaría pero con CPU kernels (muy lento)
- Mejor usar load_in_8bit que 4-bit

### Sobre la descarga de modelos

Todos los scripts usan token automático desde `/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env`

La primera descarga del modelo (~13GB) tomará:
- En red buena: 10-20 minutos
- Se cachea en `~/.cache/huggingface/hub/` después

### Sobre memoria RAM del sistema

Los scripts incluyen `-l h_vmem=10G` para reservar memoria del sistema si es necesaria.

---

## ✅ Conclusión

1. **PRIMERO:** Intenta con deimos o phobos (tfm.q)
2. **SI NO FUNCIONA:** Usa GTX 1080/1070 (student.q) con optimizaciones
3. **NUNCA:** Intentes con GTX 970

Los scripts creados te ayudarán a testear automáticamente cada opción.

