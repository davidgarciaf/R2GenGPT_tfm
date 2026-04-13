# ✅ INFORMACIÓN REAL DEL CLUSTER tfm.q

## 🎉 EXCELENTES NOTICIAS

La cola **tfm.q** tiene exactamente lo que necesitas:

### GPU: NVIDIA RTX 2080 Ti (2 por nodo)
- **VRAM:** 11.2 GB cada una ✅
- **Compute Capability:** 7.5 ✅ (Compatible 100% con bitsandbytes)
- **CUDA Driver:** 550.54.14 (Moderno)

### Recursos del Sistema:
- **CPU:** 16 cores (8 físicos + 2 threads)
- **RAM Total:** 62.7 GB
- **RAM Disponible:** 60.6 GB (deimos) / 50.5 GB (phobos)

---

## 🖥️ Comparativa: Tu Máquina vs Cluster

| Componente | Tu PC | Deimos | Phobos |
|-----------|--------|--------|--------|
| GPU | RTX 2080 Ti | RTX 2080 Ti ✅ | RTX 2080 Ti ✅ |
| VRAM GPU | 11.5 GB | 11.2 GB | 11.2 GB |
| Compute Cap | 7.5 | 7.5 ✅ | 7.5 ✅ |
| RAM Sistema | 16 GB | 62.7 GB ✅✅✅ | 62.7 GB ✅✅✅ |
| Load | N/A | 0.12 (bajo) ← **MEJOR** | 1.51 (medio) |
| Estado | Siempre disponible | Queue | Queue |

**CONCLUSIÓN:** Cluster tiene **GPU idéntica** + **más RAM sistema**

---

## 🚀 Recomendación

### 🟢 USA DEIMOS (tfm.q@deimos)
- Load más bajo (0.12 vs 1.51)
- Más disponible

### 🟡 Segunda opción: PHOBOS (tfm.q@phobos)
- Si deimos está ocupado

---

## 💻 Comandos para Lanzar Tu Trabajo

### Opción 1: Usar scripts predefinidos
```bash
# Test rápido
qsub -q tfm.q@deimos submit_test_cluster.sh

# Test de modelo completo
qsub -q tfm.q@deimos submit_model_cluster.sh
```

### Opción 2: Lanzar directamente tu script
```bash
# Script simple
qsub -q tfm.q@deimos tu_script.sh

# Con GPU específica
qsub -q tfm.q@deimos -l gpu=1 tu_script.sh

# Con reserva de memoria
qsub -q tfm.q@deimos -l h_vmem=10G tu_script.sh
```

### Monitorización
```bash
# Ver todos los jobs
qstat

# Ver mis jobs
qstat -u dgarcia

# Ver detalles
qstat -j <job_id>

# Ver logs en vivo
tail -f logs/model_cluster.log

# Cancelar job
qdel <job_id>
```

---

## ⚡ Velocidad Esperada

**Modelo:** Llama2-7B (4-bit cuantizado)  
**GPU:** RTX 2080 Ti  
**VRAM:** 11.2 GB

| Tarea | Velocidad |
|-------|-----------|
| Inferencia (batch=1) | 50-100 tokens/sec |
| Fine-tuning LoRA | ~10-20 samples/sec |
| Training | ~2-4 horas/epoch |

---

## 🎯 Uso Recomendado

### Para Development/Testing:
```bash
qsub -q tfm.q@deimos submit_test_cluster.sh
qsub -q tfm.q@deimos submit_tokenizer_cluster.sh
```

### Para Producción:
```bash
qsub -q tfm.q@deimos submit_model_cluster.sh
```

### Para Fine-tuning:
```bash
# Necesita más recursos - deimos es perfecto (62.7 GB RAM)
qsub -q tfm.q@deimos -l h_vmem=20G tu_finetune_script.sh
```

---

## ✅ Verificación de Compatibilidad

✅ **GPU:** RTX 2080 Ti es ideal para Llama2-7B  
✅ **VRAM:** 11.2 GB es suficiente (necesita ~8GB)  
✅ **Compute Cap:** 7.5 compatible 100% con bitsandbytes  
✅ **CUDA Driver:** 550.54.14 es moderno  
✅ **CPU:** 16 cores suficientes para preprocessing  
✅ **RAM Sistema:** 62.7 GB ideal para fine-tuning  

**CONCLUSIÓN: TODO PERFECTO** 🎉

---

## 📝 Información Técnica

### Distribución de VRAM (estimado)
```
Tokenizer:           ~500 MB
Llama2-7B 4-bit:     ~4.5 GB
Inference buffer:    ~2.0 GB
CUDA overhead:       ~1.0 GB
──────────────────────────────
TOTAL USADO:         ~8.0 GB
DISPONIBLE:          ~3.2 GB (margen seguro)
```

### Topología CPU
```
16 processors en 1 socket
8 cores físicos
2 threads por core
Topology: SCTTCTTCTTCTTCTTCTTCTTCTT
```

---

## 🔗 Relación con tu Máquina Local

Tu PC tiene una RTX 2080 Ti, perfecta para testing local.  
El cluster **TAMBIÉN** tiene RTX 2080 Ti para escalar.

Workflow recomendado:
1. **Develop local:** En tu PC (testing rápido)
2. **Scale up:** En cluster (entrenamiento a escala)

Mismo hardware, máxima flexibilidad.

---

## 📅 Fecha de Verificación

- **Fecha:** 2026-02-24
- **Deimos Load:** 0.12
- **Phobos Load:** 1.51
- **GPUs disponibles:** 4 (2 en cada nodo)

---

**LISTA PARA COMENZAR** ✨

```bash
qsub -q tfm.q@deimos submit_test_cluster.sh
```

