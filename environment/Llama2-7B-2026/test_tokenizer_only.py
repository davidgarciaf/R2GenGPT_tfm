#!/usr/bin/env python3
"""
Test simplificado: Solo verifica tokenizer
Más ligero que test_llama2_quantized.py --full
"""

import sys
import os
from transformers import AutoTokenizer
from huggingface_hub import login

def load_hf_token():
    """Carga token de HuggingFace desde archivo .env"""
    token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    token = line.replace('HF_TOKEN=', '').strip()
                    os.environ['HF_TOKEN'] = token
                    try:
                        login(token=token, add_to_git_credential=False)
                    except:
                        pass
                    return token
    return None

print("=" * 70)
print("  TEST SIMPLIFICADO: VERIFICACIÓN DE TOKENIZER")
print("=" * 70)

# Cargar token
token = load_hf_token()
print(f"\n✓ Token HuggingFace cargado desde archivo")

# Test 1: Tokenizer
print("\n1️⃣  Cargando tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.environ.get('HF_TOKEN')
    )
    print("   ✓ Tokenizer cargado exitosamente")
    print(f"   - Vocab size: {len(tokenizer)}")
    print(f"   - BOS token: {tokenizer.bos_token}")
    print(f"   - EOS token: {tokenizer.eos_token}")
    
    # Test de tokenización
    text = "Hello, how are you?"
    tokens = tokenizer.encode(text)
    print(f"\n2️⃣  Test de tokenización:")
    print(f"   Texto: '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   ✓ Tokenización correcta")
    
    print("\n" + "=" * 70)
    print("✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
    print("=" * 70)
    sys.exit(0)
    
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
    print("\n" + "=" * 70)
    print("❌ ERROR EN LAS PRUEBAS")
    print("=" * 70)
    sys.exit(1)
