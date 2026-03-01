#!/usr/bin/env python3
"""
Test de Llama2-7B Cuantizado (4-bit)

Este script verifica que el entorno está correctamente configurado
y que el modelo Llama2-7B puede cargar con cuantización 4-bit.

Uso:
    python3 test_llama2_quantized.py
    python3 test_llama2_quantized.py --full  # Test completo con inferencia
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

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
                    # Registrar token globalmente en HuggingFace Hub
                    try:
                        login(token=token, add_to_git_credential=False)
                    except:
                        pass  # Ignorar errores de login (puede que ya esté configurado)
                    return token
    return None

def print_section(title):
    """Imprime un encabezado de sección"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_environment():
    """Verifica que el entorno está correctamente configurado"""
    print_section("1. VERIFICACIÓN DEL ENTORNO")
    
    errors = []
    
    # PyTorch
    print(f"✓ PyTorch: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA disponible: True")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ CuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        errors.append("CUDA no está disponible")
        print(f"✗ CUDA disponible: False")
    
    # Transformers
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        errors.append(f"Transformers: {str(e)}")
        print(f"✗ Transformers: {str(e)}")
    
    # PEFT
    try:
        import peft
        print(f"✓ PEFT: {peft.__version__}")
    except ImportError as e:
        errors.append(f"PEFT: {str(e)}")
        print(f"✗ PEFT: {str(e)}")
    
    # bitsandbytes
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes: Instalado")
    except ImportError as e:
        errors.append(f"bitsandbytes: {str(e)}")
        print(f"✗ bitsandbytes: {str(e)}")
    
    # Accelerate
    try:
        import accelerate
        print(f"✓ Accelerate: {accelerate.__version__}")
    except ImportError as e:
        errors.append(f"Accelerate: {str(e)}")
        print(f"✗ Accelerate: {str(e)}")
    
    return len(errors) == 0, errors

def test_quantization_config():
    """Verifica que la configuración de cuantización es correcta"""
    print_section("2. VERIFICACIÓN DE CONFIGURACIÓN DE CUANTIZACIÓN")
    
    try:
        # Crear configuración de cuantización 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        print("✓ Configuración de cuantización 4-bit creada:")
        print(f"  - load_in_4bit: True")
        print(f"  - bnb_4bit_compute_dtype: float16")
        print(f"  - bnb_4bit_use_double_quant: True")
        print(f"  - bnb_4bit_quant_type: nf4")
        
        return True, quantization_config
    except Exception as e:
        print(f"✗ Error creando configuración: {str(e)}")
        return False, None

def test_tokenizer_load():
    """Verifica que el tokenizer puede cargarse"""
    print_section("3. CARGA DEL TOKENIZER")
    
    model_id = "meta-llama/Llama-2-7b-hf"
    
    try:
        print(f"Cargando tokenizer: {model_id}")
        print("  (Usando token de HuggingFace desde archivo .env)")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get('HF_TOKEN'))
            print(f"✓ Tokenizer cargado exitosamente")
            print(f"  - Vocab size: {len(tokenizer)}")
            print(f"  - BOS token: {tokenizer.bos_token}")
            print(f"  - EOS token: {tokenizer.eos_token}")
            return True, tokenizer
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                print(f"⚠ No se puede descargar (requiere acceso):")
                print(f"  Ejecuta: huggingface-cli login")
                print(f"  Luego intenta nuevamente")
            else:
                print(f"✗ Error: {str(e)}")
            return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, None

def test_model_load(quantization_config):
    """Verifica que el modelo puede cargarse con cuantización"""
    print_section("4. CARGA DEL MODELO (CUANTIZADO)")
    
    model_id = "meta-llama/Llama-2-7b-hf"
    
    try:
        print(f"Cargando modelo: {model_id}")
        print(f"  - Cuantización: 4-bit")
        print(f"  - Device map: auto")
        print("  (Nota: primera descarga puede tomar 10-20 minutos)")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            print(f"✓ Modelo cargado exitosamente")
            print(f"  - Tipo: {type(model).__name__}")
            print(f"  - Device: {next(model.parameters()).device}")
            print(f"  - Dtype: {next(model.parameters()).dtype}")
            
            return True, model
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                print(f"⚠ No se puede descargar (requiere acceso):")
                print(f"  Ejecuta: huggingface-cli login")
                print(f"  Luego intenta nuevamente")
            else:
                print(f"✗ Error: {str(e)}")
            return False, None
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False, None

def test_inference(model, tokenizer):
    """Verifica que la inferencia funciona"""
    print_section("5. PRUEBA DE INFERENCIA")
    
    try:
        # Prompt de prueba
        prompt = "Hello, how are you? I am"
        
        print(f"Prompt: '{prompt}'")
        print("\nGenerando respuesta...")
        
        # Tokenizar y mover a dispositivo
        # Algunos tokenizers devuelven `token_type_ids`, que los modelos
        # de causal LM como Llama no aceptan en `generate()`. Si los dejamos
        # en el diccionario, `generate` validará y fallará con un
        # ValueError similar al que vimos anteriormente.
        inputs = tokenizer(prompt, return_tensors="pt")
        # limpiar claves no utilizadas
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        # mover solo los tensores válidos a CUDA
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generar texto
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decodificar
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n✓ Inferencia completada exitosamente:")
        print(f"\nRespuesta:")
        print(f"  {response}")
        
        return True
    except Exception as e:
        print(f"✗ Error durante inferencia: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Cargar token de HuggingFace PRIMERO
    token = load_hf_token()
    if token:
        print(f"✓ Token HuggingFace cargado desde archivo\n")
        # Asegurar que el token está disponible globalmente
        if not os.environ.get('HF_TOKEN'):
            os.environ['HF_TOKEN'] = token
    else:
        print("⚠️  Token HuggingFace no encontrado\n")
    
    parser = argparse.ArgumentParser(
        description="Test de Llama2-7B Cuantizado (4-bit)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ejecutar test completo incluyendo carga de modelo e inferencia"
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-2-7b-hf",
        help="ID del modelo a usar (default: meta-llama/Llama-2-7b-hf)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  TEST DE LLAMA2-7B CUANTIZADO (4-BIT)")
    print("=" * 70)
    
    # Test 1: Entorno
    env_ok, env_errors = test_environment()
    
    if not env_ok:
        print_section("RESULTADO FINAL")
        print("✗ El entorno no está completamente configurado")
        print("\nErrores encontrados:")
        for error in env_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Test 2: Configuración
    config_ok, quantization_config = test_quantization_config()
    
    if not config_ok:
        print_section("RESULTADO FINAL")
        print("✗ No se pudo crear la configuración de cuantización")
        sys.exit(1)
    
    if not args.full:
        print_section("RESULTADO FINAL")
        print("✅ ENTORNO Y CONFIGURACIÓN: OK")
        print("\nPara ejecutar prueba completa (con descarga de modelo):")
        print("  python3 test_llama2_quantized.py --full")
        print("\nNota: El token HuggingFace está configurado automáticamente")
        return
    
    # Test 3: Tokenizer (requiere acceso a HuggingFace)
    print("\n⚠️  NOTA: Los siguientes tests requieren acceso a HuggingFace")
    print("   Token HuggingFace configurado automáticamente desde archivo .env")
    
    tokenizer_ok, tokenizer = test_tokenizer_load()
    
    if not tokenizer_ok:
        print_section("RESULTADO FINAL")
        print("⚠️  No se pudo cargar el tokenizer")
        print("   Verifica que el token esté configurado correctamente:")
        print("   python3 setup_hf_token.py")
        sys.exit(1)
    
    # Test 4: Modelo
    model_ok, model = test_model_load(quantization_config)
    
    if not model_ok:
        print_section("RESULTADO FINAL")
        print("⚠️  No se pudo cargar el modelo")
        print("   Verifica que el token esté configurado correctamente:")
        print("   python3 setup_hf_token.py")
        sys.exit(1)
    
    # Test 5: Inferencia
    inference_ok = test_inference(model, tokenizer)
    
    # Resultado final
    print_section("RESULTADO FINAL")
    if inference_ok:
        print("✅ TODOS LOS TESTS PASADOS EXITOSAMENTE")
        print("\nEl entorno está listo para usar Llama2-7B cuantizado")
    else:
        print("⚠️  Algunos tests fallaron")
        sys.exit(1)

if __name__ == "__main__":
    main()
