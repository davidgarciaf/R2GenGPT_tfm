#!/bin/bash

#$ -cwd
#$ -V
#$ -j y
#$ -o logs/model_optimized_cluster.log
# Request more host memory to avoid OS allocation failures during shard loading
# Increased from 15G to 56G because model checkpoint loading can require high
# temporary CPU memory during decompression/assembly.
#$ -l h_vmem=56G

# Script optimizado para cargar modelo con menor consumo de memoria
# Uso: qsub -q tfm.q@deimos submit_model_optimized_cluster.sh

source /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026/venv/bin/activate

echo "=============================================================================="
echo "  CARGA DE MODELO EN CLUSTER (OPTIMIZADO)"
echo "=============================================================================="
echo ""
echo "Nodo: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""
echo "Memory disponible:"
free -h | head -2
echo ""

cd /mnt/sd5/users/dgarcia/R2GenGPT/environment/Llama2-7B-2026

# Test de carga de modelo optimizado
python3 << 'EOF'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

print("=" * 70)
print("  TEST DE CARGA DE MODELO (OPTIMIZADO)")
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

# Configurar cuantización SIN double_quant para usar menos memoria
print("\n1. Configurando cuantización 4-bit (optimizado)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,  # Desactivado para menos memoria temporal
    bnb_4bit_quant_type="nf4",
)
print("   ✓ Configuración lista (sin double_quant)")

# Cargar tokenizer primero (es ligero)
print("\n2. Cargando tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.environ.get('HF_TOKEN')
    )
    print(f"   ✓ Tokenizer cargado (vocab: {len(tokenizer)})")
except Exception as e:
    print(f"   ✗ Error cargando tokenizer: {e}")
    exit(1)

# Cargar modelo con manejo de memoria optimizado
print("\n3. Cargando modelo Llama2-7B...")
print("   (Primera carga puede tomar 5-15 minutos)")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # Usar menos memoria CPU durante carga
    )
    print("   ✓ Modelo cargado exitosamente")
    print(f"   - Parámetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # Obtener info de dónde está cargado
    first_param = next(model.parameters())
    print(f"   - Device: {first_param.device}")
    print(f"   - Dtype: {first_param.dtype}")
    
    # Test de inferencia
    print("\n4. Test de inferencia...")
    prompt = "Machine learning is"
    print(f"   Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move only the tensors that the model expects to CUDA and remove unused keys
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    print("   Generando respuesta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✓ Respuesta generada:")
    print(f"\n   '{response}'")

    # Estadísticas finales
    print("\n5. Estadísticas finales:")
    try:
        used = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[1]) / 1e9
        avail = torch.cuda.mem_get_info()[0] / 1e9
        print(f"   - GPU VRAM usado: {used:.1f} GB")
        print(f"   - GPU VRAM disponible: {avail:.1f} GB")
    except Exception:
        print("   - GPU VRAM stats: unavailable")
    
    print("\n" + "=" * 70)
    print("✅ PRUEBA EXITOSA - Modelo funciona correctamente")
    print("=" * 70)
    
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    print(f"\n⚠️  Error al cargar modelo")
EOF

echo ""
echo "=============================================================================="
echo "  TEST COMPLETADO"
echo "=============================================================================="
