#!/usr/bin/env python3
"""
Configurador de Token HuggingFace

Este script configura el token de HuggingFace para usar con Llama2-7B
"""

import os
import sys

def setup_hf_token():
    """Configura el token de HuggingFace"""
    token_file = "/mnt/sd5/users/dgarcia/R2GenGPT/.hf_token_llama2.env"
    
    print("\n" + "="*70)
    print("  Configurador de Token HuggingFace")
    print("="*70 + "\n")
    
    # Leer token del archivo
    if not os.path.exists(token_file):
        print(f"❌ No se encontró archivo: {token_file}")
        sys.exit(1)
    
    try:
        with open(token_file, 'r') as f:
            content = f.read().strip()
            if content.startswith('HF_TOKEN='):
                token = content.replace('HF_TOKEN=', '').strip()
            else:
                token = content.strip()
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        sys.exit(1)
    
    if not token:
        print("❌ Token vacío en archivo")
        sys.exit(1)
    
    print(f"✓ Token encontrado: {token[:10]}...{token[-10:]}\n")
    
    # Configurar en HuggingFace Hub
    print("Configurando token en HuggingFace Hub...\n")
    
    from huggingface_hub import login
    
    try:
        login(token=token, add_to_git_credential=False)
        print("\n✓ Token configurado exitosamente en HuggingFace Hub")
        print(f"✓ Archivo de configuración: ~/.cache/huggingface/token")
        return True
    except Exception as e:
        print(f"❌ Error al configurar token: {e}")
        return False

def verify_token():
    """Verifica que el token está configurado correctamente"""
    print("\n" + "="*70)
    print("  Verificación del Token")
    print("="*70 + "\n")
    
    from huggingface_hub import whoami, model_info
    
    try:
        # Verificar que estamos autenticados
        user_info = whoami()
        print(f"✓ Usuario autenticado: {user_info['name']}")
        print(f"✓ Tipo de cuenta: {user_info.get('type', 'unknown')}")
        
        # Intentar acceder al modelo Llama2-7b
        print("\nVerificando acceso a meta-llama/Llama-2-7b...")
        try:
            info = model_info("meta-llama/Llama-2-7b")
            print(f"✓ Acceso a modelo confirmado")
            print(f"✓ Modelo: {info.modelId}")
            print(f"✓ Última actualización: {info.lastModified}")
            return True
        except Exception as e:
            print(f"⚠️  No se puede acceder al modelo: {e}")
            print("   Asegúrate de:")
            print("   1. Aceptar términos en https://huggingface.co/meta-llama/Llama-2-7b")
            print("   2. Tu token tiene permisos de lectura")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_token()
    else:
        success = setup_hf_token()
        if success:
            verify_token()
        sys.exit(0 if success else 1)
