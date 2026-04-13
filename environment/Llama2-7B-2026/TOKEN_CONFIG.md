# 🔐 Configuración de Token HuggingFace

## Resumen

El entorno Llama2-7B cuantizado está configurado para usar automáticamente el token HuggingFace desde el archivo `.hf_token_llama2.env` sin necesidad de ejecutar `huggingface-cli login` manualmente.

---

## 📋 Configuración Actual

### Ubicación del Token
```
/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env
```

### Cómo Funciona

Todos los scripts Python cargados en la carpeta Llama2-7B-2026 incluyen una función `load_hf_token()` que:

1. Lee el archivo `.hf_token_llama2.env`
2. Extrae el token HuggingFace
3. Lo establece como variable de entorno `HF_TOKEN`
4. Usa el token automáticamente al cargar modelos

### Scripts Actualizados

Los siguientes scripts ya están configurados para usar el token automáticamente:

- ✅ `test_llama2_quantized.py`
- ✅ `quick_start.py`
- ✅ `example_usage.py`
- ✅ `setup_hf_token.py` (nuevo)

---

## 🚀 Primer Uso

### Paso 1: Configurar Token (una sola vez)

```bash
cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026
source venv/bin/activate
python3 setup_hf_token.py
```

**Resultado esperado:**
```
✓ Token configurado exitosamente en HuggingFace Hub
✓ Usuario autenticado: DavidGF03
✓ Acceso a modelo confirmado
```

### Paso 2: Usar el Token Automáticamente

**Ya no es necesario `huggingface-cli login`**

Los scripts cargarán el token automáticamente:

```bash
# Test básico
python3 test_llama2_quantized.py

# Test completo con descarga de modelo
python3 test_llama2_quantized.py --full

# Demo rápida
python3 quick_start.py
```

---

## 📝 Cómo Integrar en tu Código

Si creas tus propios scripts, puedes copiar la función `load_hf_token()`:

```python
import os

def load_hf_token():
    """Carga el token de HuggingFace desde archivo .env"""
    token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    token = line.replace('HF_TOKEN=', '').strip()
                    os.environ['HF_TOKEN'] = token
                    return token
    return None

# En tu código
if __name__ == "__main__":
    token = load_hf_token()
    
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        quantization_config=quantization_config,
        device_map="auto",
        token=os.environ.get('HF_TOKEN'),  # Usar token automático
    )
```

---

## 🔍 Verificación

### Ver Estado del Token

```bash
python3 setup_hf_token.py --verify
```

### Verificar Token Manualmente

```bash
python3 -c "
import os
token_file = '/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env'
with open(token_file) as f:
    print(f.read())
"
```

### Verificar Acceso a Modelo

```bash
python3 -c "
from huggingface_hub import model_info
info = model_info('meta-llama/Llama-2-7b')
print(f'Modelo: {info.modelId}')
print(f'Última actualización: {info.lastModified}')
"
```

---

## ⚙️ Cambiar Token

Si necesitas usar otro token:

### Opción 1: Actualizar archivo .env

```bash
# Editar el archivo
nano /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env

# Cambiar la línea HF_TOKEN a tu nuevo token
HF_TOKEN=hf_tu_nuevo_token_aqui
```

### Opción 2: Actualizar desde Python

```bash
python3 << 'EOF'
token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
nuevo_token = "hf_tu_nuevo_token_aqui"

with open(token_file, 'w') as f:
    f.write(f"HF_TOKEN={nuevo_token}")

print(f"Token actualizado en {token_file}")
EOF
```

### Opción 3: Usar Variable de Entorno

```bash
export HF_TOKEN=hf_tu_token_aqui
python3 tu_script.py
```

---

## 🔐 Seguridad

### Proteger el Token

El archivo `.hf_token_llama2.env` contiene tu token de HuggingFace. Es importante:

1. **No compartir el archivo**
2. **No commitear a git**
3. **Usar permisos de archivo restrictivos**

```bash
# Revisar permisos
ls -la /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env

# Restringir acceso (si es necesario)
chmod 600 /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env
```

### Revocar Token

Si el token se ve comprometido:

1. Ir a https://huggingface.co/settings/tokens
2. Eliminar el token comprometido
3. Crear uno nuevo
4. Actualizar el archivo `.hf_token_llama2.env`

---

## 🆘 Troubleshooting

### Error: "401 Unauthorized"

Significa que el token no es válido o no tiene permisos. Soluciones:

```bash
# 1. Verificar token
python3 setup_hf_token.py

# 2. Crear nuevo token en https://huggingface.co/settings/tokens

# 3. Actualizar archivo
echo "HF_TOKEN=hf_tu_nuevo_token" > /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env

# 4. Usar script
python3 setup_hf_token.py
```

### Error: "does not appear to have a file named config.json"

Significa que no tienes acceso al modelo. Asegúrate de:

1. **Aceptar términos** en https://huggingface.co/meta-llama/Llama-2-7b
2. **Token tiene permisos de lectura**
3. **Token está actualizado**

```bash
python3 setup_hf_token.py --verify
```

### Error: "Token not found"

El archivo `.hf_token_llama2.env` no existe o está vacío:

```bash
# Verificar
cat /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env

# Si no existe, crear con formato correcto
echo "HF_TOKEN=hf_tu_token_aqui" > /mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env
```

---

## 📚 Referencias

- **HuggingFace API**: https://huggingface.co/docs/huggingface_hub/
- **Llama2-7b**: https://huggingface.co/meta-llama/Llama-2-7b
- **Crear token**: https://huggingface.co/settings/tokens

---

## ✅ Checklist

- [x] Archivo `.hf_token_llama2.env` existe
- [x] Token es válido y activo
- [x] Token tiene acceso a meta-llama/Llama-2-7b
- [x] Scripts configurados para cargarlo automáticamente
- [x] `setup_hf_token.py` creado para verificar
- [x] Documentación actualizada

---

**Última actualización**: 2026-02-19  
**Estado**: ✅ Configurado y verificado  
**Usuario**: DavidGF03  
**Modelo**: meta-llama/Llama-2-7b
