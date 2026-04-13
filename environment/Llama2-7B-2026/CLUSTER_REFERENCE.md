# 📑 ÍNDICE DE REFERENCIA RÁPIDA

## 🎯 ¿Qué debo hacer ahora?

### Opción 1: Soy nuevo (primeros pasos)
1. Lee: `CLUSTER_QUICK_START.md`
2. Ejecuta: `ssh deimos 'nvidia-smi'`
3. Lanza: `qsub -q tfm.q@deimos submit_test_cluster.sh`
4. Verifica: `tail -f logs/test_cluster.log`

### Opción 2: Necesito entender qué GPU tengo
1. Lee: `CLUSTER_ANALYSIS.md`
2. Ejecuta: `bash check_cluster_resources.sh`
3. Ejecuta: `bash check_cluster_nodes.sh`

### Opción 3: Voy a usar GTX 1080/1070
1. Lee: `GTX1080_OPTIMIZATION.md`
2. Lanza: `qsub -q student.q@pcgtx1080 submit_model_cluster.sh`
3. Si falla, aplica soluciones de `GTX1080_OPTIMIZATION.md`

### Opción 4: Necesito información técnica
1. `CLUSTER_ANALYSIS.md` - Análisis técnico detallado
2. `TEST_REPORT.md` - Resultados de pruebas
3. `VERIFICATION_REPORT.md` - Verificación del entorno

---

## 📋 Archivos por Propósito

### Empezar Rápido
- `CLUSTER_QUICK_START.md` - Guía de 5 pasos

### Entender los Recursos
- `CLUSTER_ANALYSIS.md` - Análisis completo
- `check_cluster_resources.sh` - Ver en terminal

### Lanzar Trabajos
- `submit_test_cluster.sh` - Test rápido
- `submit_tokenizer_cluster.sh` - Solo tokenizer
- `submit_model_cluster.sh` - Modelo completo

### Solucionar Problemas (GTX 1080)
- `GTX1080_OPTIMIZATION.md` - 4 soluciones diferentes

### Información de Instalación Local
- `README.md` - Guía general
- `TEST_REPORT.md` - Resultados de pruebas
- `VERIFICATION_REPORT.md` - Verificación completa

---

## 🚀 Comandos más Usados

```bash
# VER QUÉ GPUs HAY
ssh deimos 'nvidia-smi'
ssh phobos 'nvidia-smi'

# LANZAR TEST
qsub -q tfm.q@deimos submit_test_cluster.sh
qsub -q student.q@pcgtx1080 submit_model_cluster.sh

# VER ESTADO
qstat
qstat -u $USER

# VER LOGS EN TIEMPO REAL
tail -f logs/test_cluster.log
tail -f logs/model_cluster.log

# CANCELAR JOB
qdel <job_id>

# CREAR CARPETA DE LOGS (si no existe)
mkdir -p logs
```

---

## 💡 Recomendaciones por Scenario

### Scenario A: Tengo acceso a deimos/phobos con GPU moderna
```bash
# Simplemente usa:
qsub -q tfm.q@deimos submit_model_cluster.sh

# Lee CLUSTER_QUICK_START.md para detalles
```

### Scenario B: Solo tengo student.q (GTX 1080/1070)
```bash
# Intenta directamente:
qsub -q student.q@pcgtx1080 submit_model_cluster.sh

# Si falla, lee GTX1080_OPTIMIZATION.md
```

### Scenario C: No sé qué GPU tengo
```bash
# Ejecuta para ver:
bash check_cluster_resources.sh
bash check_cluster_nodes.sh

# Luego lee CLUSTER_ANALYSIS.md
```

---

## 📊 Tabla de Decisión Rápida

| GPU | VRAM | Compute Cap | Funciona | Recomendación |
|-----|------|-------------|----------|---------------|
| V100/A100 | 32GB | 7.0/8.0 | ✅ Sí | 🟢 INTENTA |
| RTX 2080 Ti | 11GB | 7.5 | ✅ Sí | 🟢 INTENTA |
| GTX 1080 | 8GB | 6.1 | 🟡 Quizás | 🟡 con optimizaciones |
| GTX 1070 | 8GB | 6.1 | 🟡 Quizás | 🟡 con optimizaciones |
| GTX 970 | 4GB | 5.2 | ❌ No | 🔴 EVITA |

---

## ⚡ Quick Troubleshooting

**P: ¿Cómo sé qué GPU tiene deimos?**
```bash
ssh deimos 'nvidia-smi'
```

**P: ¿Cómo cancelo un job?**
```bash
qdel <job_id>  # Reemplaza <job_id> con el número de job
```

**P: ¿Dónde están los logs?**
```bash
ls logs/
tail -f logs/model_cluster.log
```

**P: ¿Cómo veo si mi job está corriendo?**
```bash
qstat
```

**P: ¿El modelo se descarga automáticamente?**
```
Sí, automáticamente en primera ejecución (~13GB, 10-20 min)
```

**P: ¿Necesito hacer setup adicional en el cluster?**
```
No, los scripts se encargan de todo automáticamente
```

---

## 🎓 Orden Recomendado de Lectura

1. **Primero** (5 min): `CLUSTER_QUICK_START.md`
2. **Luego** (10 min): `bash check_cluster_resources.sh`
3. **Si quieres detalles** (20 min): `CLUSTER_ANALYSIS.md`
4. **Si usas GTX 1080** (15 min): `GTX1080_OPTIMIZATION.md`
5. **Para troubleshooting**: Este archivo (`CLUSTER_REFERENCE.md`)

---

## 📁 Ubicación Base

Todos los archivos están en:
```
/mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/
```

Copiar el path completo en comandos qsub:
```bash
qsub -q tfm.q@deimos /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/submit_test_cluster.sh
```

---

## 🆘 Obtener Ayuda

1. **Para entender recursos**: Leer `CLUSTER_ANALYSIS.md`
2. **Para problemas GTX 1080**: Leer `GTX1080_OPTIMIZATION.md`
3. **Para errores en logs**: Ver `tail -f logs/*.log`
4. **Para dudas técnicas**: Revisar `VERIFICATION_REPORT.md`

---

## ✅ Checklist Final

Antes de lanzar tu trabajo productivo:

- [ ] Leí `CLUSTER_QUICK_START.md`
- [ ] Ejecuté `ssh deimos 'nvidia-smi'`
- [ ] Lancé al menos un test exitoso
- [ ] Verifiqué que los logs aparecen correctamente
- [ ] Si uso GTX 1080, leí las soluciones de optimización
- [ ] Mi token HuggingFace está en `/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env`

---

**Última actualización:** 2026-02-24
**Versión:** 1.0
**Estado:** Completo y listo para usar ✅

