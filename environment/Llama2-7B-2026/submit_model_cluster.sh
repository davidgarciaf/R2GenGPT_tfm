#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/model_cluster.log
#$ -l h_vmem=10G

# Script para cargar modelo completo en cluster
# Uso: qsub -q tfm.q@deimos submit_model_cluster.sh

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "=============================================================================="
echo "  CARGA DE MODELO EN CLUSTER"
echo "=============================================================================="
echo ""
echo "Nodo: $(hostname)"
echo "GPU Info:"
nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""
echo "Memory disponible:"
free -h | head -2
echo ""

cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

# Test completo con modelo
python3 << 'EOF'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

print("=" * 70)
print("  TEST DE CARGA DE MODELO")
print("=" * 70)

# Cargar token
token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
with open(token_file, 'r') as f:
    for line in f:
        if line.startswith('HF_TOKEN='):
            token = line.replace('HF_TOKEN=', '').strip()
            os.environ['HF_TOKEN'] = token
            login(token=token, add_to_git_credential=False)

print(f"\n✓ Token HuggingFace cargado")

# Verificar GPU
print(f"\n✓ GPU disponible: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✓ GPU VRAM libre: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

# Configurar cuantización
print("\n1. Configurando cuantización 4-bit...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
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
print("\n3. Cargando modelo Llama2-7B con cuantización 4-bit...")
print("   (Esta operación puede tomar 2-5 minutos)")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("   ✓ Modelo cargado exitosamente")
    print(f"   - Parámetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")
    
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
    print(f"   Prompt: '{prompt}'")
    print(f"   Respuesta: '{response}'")
    print("   ✓ Inferencia completada")
    
    print("\n" + "=" * 70)
    print("✅ PRUEBA EXITOSA - Modelo funciona correctamente en este nodo")
    print("=" * 70)
    
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    print(f"\n⚠️  Este nodo podría no ser óptimo para este modelo")
    print(f"   Considera usar un nodo con más VRAM o GPU más moderna")
EOF

echo ""
echo "=============================================================================="
echo "  TEST COMPLETADO"
echo "=============================================================================="
