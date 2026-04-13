# 🎓 ÍNDICE MAESTRO - ANÁLISIS DE CLUSTER

## 📚 Documentación Completa sobre Cluster

### Archivo | Propósito | Tamaño | Tiempo Lectura
-----------|-----------|--------|---------------
**CLUSTER_QUICK_START.md** | Guía rápida de 5 pasos para comenzar | 5.1 KB | ⭐ 5 min
**CLUSTER_ANALYSIS.md** | Análisis técnico detallado de compatibilidad | 6.6 KB | ⭐⭐⭐ 15 min
**CLUSTER_REFERENCE.md** | Referencia rápida de comandos y decisiones | 5.0 KB | ⭐ 5 min
**GTX1080_OPTIMIZATION.md** | Soluciones para GTX 1080/1070 | 6.6 KB | ⭐⭐ 10 min
**check_cluster_resources.sh** | Script que muestra análisis de recursos | 5.5 KB | 2 min (ejecución)
**check_cluster_nodes.sh** | Verifica GPUs actuales en cluster | 2.3 KB | 1 min (ejecución)

---

## 🔧 Scripts para Cluster

### Script | Propósito | Uso
---------|-----------|-----
**submit_test_cluster.sh** | Test rápido del entorno | `qsub -q tfm.q@deimos submit_test_cluster.sh`
**submit_tokenizer_cluster.sh** | Test de tokenizer (muy ligero) | `qsub -q tfm.q@deimos submit_tokenizer_cluster.sh`
**submit_model_cluster.sh** | Test de modelo completo + inferencia | `qsub -q tfm.q@deimos submit_model_cluster.sh`

---

## 🎯 ¿Qué Archivo Necesito?

### Si soy principiante:
1. **Empezar:** CLUSTER_QUICK_START.md
2. **Entender:** Ejecutar `bash check_cluster_resources.sh`
3. **Lanzar:** Usar `submit_test_cluster.sh`

### Si necesito decisiones técnicas:
1. **Análisis:** CLUSTER_ANALYSIS.md
2. **Referencia:** CLUSTER_REFERENCE.md
3. **Troubleshooting:** GTX1080_OPTIMIZATION.md

### Si tengo dudas:
1. **¿Qué GPU hay?:** `ssh deimos 'nvidia-smi'`
2. **¿Cómo decido?:** CLUSTER_ANALYSIS.md (tabla comparativa)
3. **¿Cómo lanzo?:** CLUSTER_QUICK_START.md (pasos)

### Si tengo problema:
1. **Ver logs:** `tail -f logs/model_cluster.log`
2. **Entender el error:** CLUSTER_ANALYSIS.md
3. **Soluciones GTX 1080:** GTX1080_OPTIMIZATION.md

---

## 📊 Recomendaciones Rápidas

| GPU | Compatibilidad | Acción |
|-----|-----------------|--------|
| **V100/A100** | ✅ Óptima | Usa directamente: `qsub -q tfm.q@deimos submit_model_cluster.sh` |
| **RTX 2080 Ti** | ✅ Excelente | Usa directamente: `qsub -q tfm.q@deimos submit_model_cluster.sh` |
| **RTX series** | ✅ Buena | Usa directamente: `qsub -q tfm.q@phobos submit_model_cluster.sh` |
| **GTX 1080** | 🟡 Marginal | Lee GTX1080_OPTIMIZATION.md y aplica soluciones |
| **GTX 1070** | 🟡 Marginal | Lee GTX1080_OPTIMIZATION.md y aplica soluciones |
| **GTX 970** | ❌ No funciona | Usa Ollama/LLaMA.cpp o solicita otro nodo |

---

## 🚀 Quick Start (30 segundos)

1. Lee: **CLUSTER_QUICK_START.md**
2. Lanza: `qsub -q tfm.q@deimos submit_test_cluster.sh`
3. Espera: `tail -f logs/test_cluster.log`

---

## 📋 Archivos por Categoría

### Análisis y Planificación
- `CLUSTER_ANALYSIS.md` - Análisis técnico completo
- `CLUSTER_REFERENCE.md` - Guía de referencia rápida
- `check_cluster_resources.sh` - Verificar recursos localmente
- `check_cluster_nodes.sh` - Verificar nodos en vivo

### Ejecución y Testing
- `submit_test_cluster.sh` - Test del entorno
- `submit_tokenizer_cluster.sh` - Test del tokenizer
- `submit_model_cluster.sh` - Test del modelo completo

### Soluciones y Troubleshooting
- `GTX1080_OPTIMIZATION.md` - Soluciones para Maxwell
- `CLUSTER_QUICK_START.md` - Pasos iniciales

---

## 💡 Flujo Recomendado

```
Principiante
    ↓
Lee CLUSTER_QUICK_START.md (5 min)
    ↓
Ejecuta: ssh deimos 'nvidia-smi'
    ↓
Lanza: qsub -q tfm.q@deimos submit_test_cluster.sh
    ↓
Espera resultados
    ↓
¿Funcionó?
    ├─ Sí → Lanza tu trabajo productivo ✅
    └─ No → Lee CLUSTER_ANALYSIS.md + GTX1080_OPTIMIZATION.md
```

---

## 📍 Ubicación de Archivos

Todos en: `/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/`

```
Llama2-7B-2026/
├── CLUSTER_QUICK_START.md          ← Empieza aquí
├── CLUSTER_ANALYSIS.md
├── CLUSTER_REFERENCE.md
├── CLUSTER_INDEX.md                ← Este archivo
├── GTX1080_OPTIMIZATION.md
├── submit_test_cluster.sh
├── submit_tokenizer_cluster.sh
├── submit_model_cluster.sh
├── check_cluster_resources.sh
├── check_cluster_nodes.sh
├── test_llama2_quantized.py        (tests locales)
├── test_tokenizer_only.py
├── quick_start.py
├── example_usage.py
├── logs/                           (se crea al lanzar jobs)
└── venv/                           (entorno virtual)
```

---

## ✅ Checklist de Setup

- [ ] Leí CLUSTER_QUICK_START.md
- [ ] Ejecuté `ssh deimos 'nvidia-smi'`
- [ ] Creé carpeta logs: `mkdir -p logs`
- [ ] Lancé test inicial: `qsub -q tfm.q@deimos submit_test_cluster.sh`
- [ ] Vi los resultados: `tail -f logs/test_cluster.log`
- [ ] Leí las soluciones si usaré GTX 1080: GTX1080_OPTIMIZATION.md

---

## 🎓 Orden de Lectura Recomendado

1. **Este archivo (1 min)** - Entender la estructura
2. **CLUSTER_QUICK_START.md (5 min)** - Ver pasos iniciales
3. **Ejecutar tests (5-10 min)** - Verificar que funciona
4. **CLUSTER_ANALYSIS.md (15 min)** - Entender detalles si necesitas
5. **GTX1080_OPTIMIZATION.md (10 min)** - Solo si usas Maxwell

**Tiempo total:** 30-40 minutos para estar completamente listo

---

## 🔗 Relaciones entre Archivos

```
CLUSTER_INDEX.md (este archivo)
    ├─ CLUSTER_QUICK_START.md (primer paso)
    │   ├─ submit_test_cluster.sh
    │   ├─ submit_tokenizer_cluster.sh
    │   └─ submit_model_cluster.sh
    │
    ├─ CLUSTER_ANALYSIS.md (análisis detallado)
    │   ├─ check_cluster_resources.sh
    │   └─ check_cluster_nodes.sh
    │
    ├─ CLUSTER_REFERENCE.md (referencia rápida)
    │
    └─ GTX1080_OPTIMIZATION.md (si necesitas)
```

---

## 🎯 Decisiones Clave

### 1. ¿Qué GPU debo usar?
→ Ver tabla en **CLUSTER_QUICK_START.md**

### 2. ¿Cómo verifico disponibilidad?
→ Leer **CLUSTER_ANALYSIS.md** o ejecutar `check_cluster_nodes.sh`

### 3. ¿Cómo lanzo mi trabajo?
→ Seguir pasos en **CLUSTER_QUICK_START.md**

### 4. ¿Tengo problemas con GTX 1080?
→ Leer **GTX1080_OPTIMIZATION.md**

### 5. ¿Cómo monitorizo el progreso?
→ Ver sección "Monitorización" en **CLUSTER_REFERENCE.md**

---

## 💾 Resumen de Capacidades

✅ **Funciona bien:**
- V100, A100, H100
- RTX 2080 Ti, RTX 3090, RTX 4090
- Otros RTX modernos

🟡 **Funciona con optimizaciones:**
- GTX 1080 (8GB, CC 6.1)
- GTX 1070 (8GB, CC 6.1)

❌ **No funciona:**
- GTX 970 (4GB, CC 5.2)
- GPUs antiguas sin suficiente VRAM

---

## 📞 Soporte

Para dudas, en orden de utilidad:
1. Leer **CLUSTER_ANALYSIS.md** - Análisis completo
2. Ver **CLUSTER_REFERENCE.md** - Referencia rápida
3. Ejecutar `bash check_cluster_resources.sh` - Ver información
4. Leer logs: `tail -f logs/*.log`

---

**Última actualización:** 2026-02-24
**Versión:** 1.0
**Estado:** Listo para usar ✅

Todo está preparado para lanzar tu trabajo en el cluster. ¡Buena suerte! 🚀

