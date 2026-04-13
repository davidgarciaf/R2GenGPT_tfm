#!/usr/bin/env python3
"""
Ejemplos de uso de Llama2-7B Cuantizado (4-bit)

Este archivo contiene varios ejemplos de cómo usar el modelo Llama2-7B
con cuantización 4-bit en diferentes escenarios.

Requisitos:
    - Tener el entorno activado
    - Acceso a HuggingFace (huggingface-cli login)
    - Aceptación de términos de uso de Llama2-7b
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

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

# ============================================================================
# EJEMPLO 1: Carga básica del modelo cuantizado
# ============================================================================

def example_1_basic_load():
    """Ejemplo básico: cargar modelo cuantizado"""
    print("=" * 70)
    print("EJEMPLO 1: Carga Básica del Modelo Cuantizado")
    print("=" * 70)
    
    token = os.environ.get('HF_TOKEN')
    
    # Configuración de cuantización 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)
    
    print(f"✓ Modelo cargado: {model.config.model_type}")
    print(f"✓ Parámetros: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"✓ Dtype: {next(model.parameters()).dtype}")
    print()
    
    return model, tokenizer

# ============================================================================
# EJEMPLO 2: Inferencia simple (texto completado)
# ============================================================================

def example_2_simple_inference(model, tokenizer):
    """Ejemplo 2: Inferencia simple"""
    print("=" * 70)
    print("EJEMPLO 2: Inferencia Simple")
    print("=" * 70)
    
    prompt = "Artificial intelligence is"
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decodificar
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Prompt: {prompt}")
    print(f"Respuesta:\n{response}\n")

# ============================================================================
# EJEMPLO 3: Configurar LoRA para fine-tuning eficiente
# ============================================================================

def example_3_lora_setup(model):
    """Ejemplo 3: Configurar LoRA para fine-tuning"""
    print("=" * 70)
    print("EJEMPLO 3: Configurar LoRA para Fine-Tuning Eficiente")
    print("=" * 70)
    
    # Configuración LoRA
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,  # Alpha
        target_modules=["q_proj", "v_proj"],  # Módulos a ajustar
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Aplicar LoRA
    model = get_peft_model(model, lora_config)
    
    # Mostrar parámetros trainables
    model.print_trainable_parameters()
    print()
    
    return model

# ============================================================================
# EJEMPLO 4: Generación con diferentes parámetros
# ============================================================================

def example_4_generation_parameters(model, tokenizer):
    """Ejemplo 4: Generación con diferentes parámetros"""
    print("=" * 70)
    print("EJEMPLO 4: Generación con Diferentes Parámetros")
    print("=" * 70)
    
    prompt = "What is machine learning?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Parámetros diferentes
    generation_configs = {
        "Greedy (determinista)": {
            "max_length": 100,
            "num_beams": 1,
            "do_sample": False,
        },
        "Temperature baja (conservador)": {
            "max_length": 100,
            "num_beams": 1,
            "temperature": 0.3,
            "do_sample": True,
        },
        "Temperature alta (creativo)": {
            "max_length": 100,
            "num_beams": 1,
            "temperature": 1.5,
            "do_sample": True,
        },
        "Top-p sampling": {
            "max_length": 100,
            "num_beams": 1,
            "top_p": 0.9,
            "do_sample": True,
        }
    }
    
    with torch.no_grad():
        for config_name, config in generation_configs.items():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **config)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{config_name}:")
            print(f"  {response[:200]}...\n")

# ============================================================================
# EJEMPLO 5: Conversación simple
# ============================================================================

def example_5_conversation(model, tokenizer):
    """Ejemplo 5: Simular una conversación"""
    print("=" * 70)
    print("EJEMPLO 5: Simulación de Conversación")
    print("=" * 70)
    
    conversation_history = ""
    user_inputs = [
        "Hi, what is your name?",
        "Tell me about AI",
    ]
    
    for user_input in user_inputs:
        # Agregar input a historial
        conversation_history += f"User: {user_input}\nAssistant:"
        
        # Tokenizar
        inputs = tokenizer(conversation_history, return_tensors="pt").to("cuda")
        
        # Generar respuesta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Extraer solo la nueva respuesta
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Agregar respuesta a historial
        conversation_history += f" {response}\n"
        
        print(f"User: {user_input}")
        print(f"Assistant: {response}\n")

# ============================================================================
# EJEMPLO 6: Calcular tokens y memoria
# ============================================================================

def example_6_memory_analysis(model, tokenizer):
    """Ejemplo 6: Analizar memoria y tokens"""
    print("=" * 70)
    print("EJEMPLO 6: Análisis de Memoria y Tokens")
    print("=" * 70)
    
    # Memoria del modelo
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Memoria del modelo: {model_memory:.2f} GB")
    
    # Memoria de CUDA
    if torch.cuda.is_available():
        print(f"Memoria CUDA total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Memoria CUDA reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Memoria CUDA allocada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Análisis de tokens
    test_text = "This is a test to count tokens in a sentence."
    tokens = tokenizer(test_text)
    print(f"\nTexto: '{test_text}'")
    print(f"Número de tokens: {len(tokens['input_ids'])}")
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Cargar token de HuggingFace
    token = load_hf_token()
    
    print("\n" + "=" * 70)
    print("EJEMPLOS DE USO: LLAMA2-7B CUANTIZADO (4-BIT)")
    print("=" * 70 + "\n")
    
    if token:
        print("✓ Token HuggingFace cargado automáticamente\n")
    
    # IMPORTANTE: Descomentar los ejemplos que deseas ejecutar
    # Nota: Solo cargar el modelo una vez para ahorrar memoria
    
    try:
        # Cargar modelo y tokenizer
        print("Cargando modelo Llama2-7B cuantizado...")
        print("(Esto puede tomar 2-5 minutos en la primera ejecución)\n")
        model, tokenizer = example_1_basic_load()
        
        # Ejecutar ejemplos
        # example_2_simple_inference(model, tokenizer)
        # example_3_lora_setup(model)
        # example_4_generation_parameters(model, tokenizer)
        # example_5_conversation(model, tokenizer)
        # example_6_memory_analysis(model, tokenizer)
        
        print("\n✅ Ejemplos cargados. Descomenta los que quieras ejecutar.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de:")
        print("  1. Ejecutar: source venv/bin/activate")
        print("  2. Ejecutar: huggingface-cli login")
        print("  3. Aceptar los términos de uso de Llama2-7b")
