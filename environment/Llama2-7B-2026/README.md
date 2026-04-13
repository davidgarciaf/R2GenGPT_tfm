# 🦙 Llama2-7B Cuantizado (4-bit) - Guía de Uso

Entorno completamente configurado para ejecutar **Llama2-7B con cuantización 4-bit** usando `bitsandbytes`, `PEFT` (LoRA), y `transformers`.

---

## 📋 Contenido de la Carpeta

```
Llama2-7B-2026/
├── venv/                              # Entorno virtual (4.6 GB)
│
├── 📖 DOCUMENTACIÓN
│   ├── README.md                      # Este archivo
│   ├── INSTALL_GUIDE.md              # Guía de instalación detallada
│   ├── CONFLICT_RESOLUTION.md        # Análisis de conflictos de dependencias
│   ├── VERIFICATION_REPORT.md        # Reporte de verificación final
│
├── 🧪 PRUEBAS
│   ├── test_llama2_quantized.py      # Test del entorno y modelo
│   ├── example_usage.py               # Ejemplos de uso
│
├── 🔧 SCRIPTS
│   ├── activate.sh                   # Script de activación rápida
│   ├── install_complete.sh           # Script de instalación
│   ├── install_env.sh                # Script alternativo de instalación
│
├── 📦 DEPENDENCIAS
│   ├── requirements.txt               # Requisitos originales
│   └── requirements_quantized.txt    # Requisitos con versiones pinned
```

---

## 🚀 Inicio Rápido

### 1. Activar el Entorno

**Opción A: Usando script rápido (recomendado)**
```bash
bash /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/activate.sh
```

**Opción B: Activación manual**
```bash
source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate
```

### 2. Verificar Entorno

```bash
python3 test_llama2_quantized.py
```

**Resultado esperado:**
```
✅ ENTORNO Y CONFIGURACIÓN: OK
```

### 3. Configurar Acceso a HuggingFace

```bash
# Instalar CLI (si no está instalado)
pip install huggingface-hub

# Iniciar sesión
huggingface-cli login
```

Necesitarás:
1. Crear cuenta en https://huggingface.co/
2. Aceptar términos de uso de `meta-llama/Llama-2-7b`
3. Crear token de acceso en: https://huggingface.co/settings/tokens
4. Pegar el token cuando se solicite

---

## 📚 Ejemplos de Uso

### Test Básico (sin descargar modelo)

```bash
# Verifica que el entorno está configurado correctamente
python3 test_llama2_quantized.py
```

### Test Completo (descarga y prueba el modelo)

```bash
# Requiere: huggingface-cli login (ejecutar primero)
python3 test_llama2_quantized.py --full
```

### Ejemplo de Código

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuración de cuantización 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Cargar modelo
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto",
)

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Generar texto
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔍 Fine-tuning con LoRA

### Configuración Básica

```python
from peft import LoraConfig, TaskType, get_peft_model

# Configurar LoRA
lora_config = LoraConfig(
    r=8,                              # Rank
    lora_alpha=16,                    # Alpha
    target_modules=["q_proj", "v_proj"],  # Módulos a ajustar
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Aplicar LoRA al modelo
model = get_peft_model(model, lora_config)

# Ver parámetros trainables
model.print_trainable_parameters()
# OUTPUT: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.062
```

### Entrenamiento con Lightning

```python
import lightning as L
from torch.utils.data import DataLoader

class LLMFinetuner(L.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        return outputs.loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

# Crear trainer
trainer = L.Trainer(max_epochs=3, precision="16-mixed", devices=1)
trainer.fit(model, train_dataloaders=train_loader)
```

---

## ⚙️ Variables de Entorno Importantes

```bash
# GPU a usar (0 = primera GPU)
export CUDA_VISIBLE_DEVICES=0

# Configuración de memoria CUDA
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3000
```

---

## 📊 Especificaciones

### Hardware Requerido

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| **GPU VRAM** | 8 GB | 12+ GB |
| **RAM** | 16 GB | 32+ GB |
| **Espacio Disco** | 20 GB | 40 GB |
| **GPU** | Cualquier NVIDIA | RTX 2080 Ti+ |

### Configuración Instalada

| Componente | Versión |
|-----------|---------|
| **Python** | 3.8.10 |
| **CUDA** | 11.8 |
| **cuDNN** | 9.1.0.70 |
| **PyTorch** | 2.4.1+cu118 |
| **Transformers** | 4.30.2 |
| **PEFT** | 0.13.2 |
| **bitsandbytes** | latest |
| **Lightning** | 2.0.5 |

---

## 🆘 Troubleshooting

### Error: "CUDA out of memory"

Aumentar la configuración de memoria:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:5000
```

O reducir batch size en tu código:
```python
model.generate(..., max_length=50)  # Reducir length
```

### Error: "Module not found: bitsandbytes"

Reinstalar bitsandbytes:
```bash
source venv/bin/activate
pip install --force-reinstall bitsandbytes
```

### Error: "Unauthorized (401)" al descargar modelo

Configurar acceso a HuggingFace:
```bash
huggingface-cli login
# Ingresar token
```

Asegurarse de aceptar términos en: https://huggingface.co/meta-llama/Llama-2-7b

### Error: "transformers version mismatch"

Reinstalar versión correcta:
```bash
source venv/bin/activate
pip install transformers==4.30.2 --force-reinstall
```

---

## 📖 Lecturas Adicionales

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **bitsandbytes Quantization**: https://github.com/TimDettmers/bitsandbytes
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://github.com/huggingface/peft
- **PyTorch Lightning**: https://lightning.ai/

---

## ✅ Checklist de Configuración

- [ ] Entorno virtual activado
- [ ] Test básico ejecutado: `python3 test_llama2_quantized.py`
- [ ] HuggingFace CLI configurado: `huggingface-cli login`
- [ ] Aceptados términos de Llama2-7b en HuggingFace
- [ ] Test completo ejecutado: `python3 test_llama2_quantized.py --full`
- [ ] Memoria CUDA configurada si es necesario

---

## 🎯 Próximos Pasos

1. **Familiarizarse con el modelo**: Ejecutar ejemplos en `example_usage.py`
2. **Fine-tuning**: Usar LoRA para ajustar el modelo con tus datos
3. **Inferencia**: Integrar en tus aplicaciones
4. **Evaluación**: Usar herramientas como BLEU, ROUGE para evaluar resultados

---

**Última actualización**: 2026-02-19  
**Estado**: ✅ Listo para usar  
**Entorno**: Completamente verificado y configurado
