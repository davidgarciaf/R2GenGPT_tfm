#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/model_echo_cluster.log
#$ -l h_vmem=56G

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "=============================================================================="
echo "  PRUEBA: TOKEN VISIBLE (PARCIAL) + RESPUESTA DEL MODELO"
echo "=============================================================================="
echo "Nodo: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""
cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

python3 << 'PY'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

print('\n--- INICIANDO PRUEBA CLARA DE TOKEN Y RESPUESTA ---\n')

# Cargar token desde archivo y mostrar confirmación (parcialmente enmascarado)
token_file = '/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env'
token = None
try:
    with open(token_file, 'r') as f:
        for line in f:
            if line.startswith('HF_TOKEN='):
                token = line.replace('HF_TOKEN=', '').strip()
                break
except Exception as e:
    print(f"ERROR leyendo token file: {e}")

if not token:
    print('✗ Token no encontrado en .hf_token_llama2.env; abortando prueba')
    raise SystemExit(1)

# Mostrar token parcialmente (primeros 6 caracteres) para confirmar que se leyó
print('✓ Token leído (masked):', token[:6] + '...' + token[-4:])
os.environ['HF_TOKEN'] = token
login(token=token, add_to_git_credential=False)
print('✓ Login HuggingFace: OK')

# Configuración de cuantización (misma receta probada)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type='nf4',
)

print('\n✓ Cargando tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'))
print('   - Tokenizer cargado. Vocab size:', getattr(tokenizer, 'vocab_size', 'unknown'))

print('\n✓ Cargando modelo (optimizado, puede tardar varios minutos)...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantization_config=quantization_config,
    device_map='auto',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
print('   - Modelo cargado. Primer parámetro en device:', next(model.parameters()).device)

# Inferencia clara
prompt = "The quick test: Machine learning is"
print('\n✓ Tokenizando prompt:')
inputs = tokenizer(prompt, return_tensors='pt')
print('   - Keys from tokenizer:', list(inputs.keys()))
if 'token_type_ids' in inputs:
    print('   - Eliminando token_type_ids (no usado por este modelo)')
    del inputs['token_type_ids']
inputs = {k: v.to('cuda') for k, v in inputs.items()}

print('\n✓ Generando respuesta (mostraré la salida completa):')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\n--- RESPUESTA DEL MODELO ---')
print(response)
print('--- FIN RESPUESTA ---\n')

print('Estadísticas VRAM:')
try:
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    avail = torch.cuda.mem_get_info()[0] / 1e9
    used = total - avail
    print(f'   - Total: {total:.1f} GB, Usado aprox: {used:.1f} GB, Disponible: {avail:.1f} GB')
except Exception:
    print('   - VRAM stats not available')

print('\n=== PRUEBA COMPLETADA ===')
PY

echo "=============================================================================="
echo "  TEST ECHO COMPLETADO"
echo "=============================================================================="
