#!/usr/bin/env python3
"""
Quick Start Script - Llama2-7B Cuantizado

Ejecuta este script para una prueba rápida del modelo.

Uso:
    python3 quick_start.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Cargar token HuggingFace desde archivo .env
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

def main():
    # Cargar token de HuggingFace
    token = load_hf_token()
    
    print("\n" + "="*70)
    print("  QUICK START - LLAMA2-7B CUANTIZADO (4-BIT)")
    print("="*70 + "\n")
    
    if token:
        print("✓ Token HuggingFace cargado automáticamente\n")
    
    # Paso 1: Verificar entorno
    print("1️⃣  Verificando entorno...")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Paso 2: Configuración de cuantización
    print("\n2️⃣  Configurando cuantización 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("   ✓ Configuración lista")
    
    # Paso 3: Cargar modelo
    print("\n3️⃣  Cargando Llama2-7B...")
    print("   (Primera carga puede tomar 2-5 minutos)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("   ✓ Modelo cargado")
    except Exception as e:
        if "401" in str(e):
            print("   ✗ Error: No autorizado")
            print("\n   Solución:")
            print("   Verifica que el token en .hf_token_llama2.env es válido")
            print("   O ejecuta: huggingface-cli login")
            return
        raise
    
    # Paso 4: Cargar tokenizer
    print("\n4️⃣  Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=os.environ.get('HF_TOKEN'))
    print("   ✓ Tokenizer listo")
    
    # Paso 5: Inferencia
    print("\n5️⃣  Probando inferencia...")
    prompt = "What is artificial intelligence? AI is"
    print(f"   Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n   Respuesta:")
    print(f"   {response}")
    
    # Resumen
    print("\n" + "="*70)
    print("✅ TODO FUNCIONANDO CORRECTAMENTE")
    print("="*70)
    print("\nSiguientes pasos:")
    print("  1. Ver ejemplos: cat example_usage.py")
    print("  2. Fine-tuning con LoRA: ver INSTALL_GUIDE.md")
    print("  3. Usar en tu proyecto: importar funciones clave")
    print("\n")

if __name__ == "__main__":
    main()
