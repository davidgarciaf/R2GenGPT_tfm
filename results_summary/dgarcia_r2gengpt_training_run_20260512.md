# R2GenGPT Training Run - May 12, 2026

## Run Information

**Date**: May 12, 2026  
**Job ID**: 22522  
**Host**: deimos  
**Script**: `scripts/run_1-1.shallow_run_iuxray_rep_gpt.sh`  
**Status**: ✅ Completed successfully  
**Duration**: ~2h 15m 15s  
**Exit Code**: 0

---

## Environment Specifications

### Python & Framework Versions
- **Python**: 3.8.10
- **PyTorch**: 2.4.1+cu118
- **Transformers**: 4.46.3
- **CUDA**: 12.4
- **NVIDIA Driver**: 550.54.14
- **Total Packages**: 127

### GPU Hardware
- 2x NVIDIA GeForce RTX 2080 Ti
- VRAM per GPU: 11264 MiB
- Usage: 1 GPU (device 0)

### Models Used
- **LLM**: meta-llama/Llama-3.2-1B-Instruct
- **Vision**: microsoft/swin-base-patch4-window7-224

---

## Training Configuration

### Quantization
- `load_in_4bit`: True
- `bnb_4bit_quant_type`: nf4
- `bnb_4bit_compute_dtype`: float16
- `bnb_4bit_use_double_quant`: False

### Training Parameters
- `batch_size`: 2
- `val_batch_size`: 2
- `max_epochs`: 15
- `learning_rate`: 0.0001
- `optimizer`: AdamW
- `precision`: 16-mixed
- `devices`: 1
- `strategy`: auto
- `num_workers`: 1

### Model Configuration
- `freeze_vm`: True (vision model frozen)
- `vis_use_lora`: False
- `llm_use_lora`: False
- `max_length`: 60
- `min_new_tokens`: 40
- `max_new_tokens`: 100
- `repetition_penalty`: 2.0
- `length_penalty`: 2.0

### Model Parameters
- Trainable params: 2.1 M
- Non-trainable params: 836 M
- Total params: 838 M
- Estimated size: 3,352.487 MB

---

## Dataset

- **Name**: iu_xray
- **Annotation**: `/mnt/sd5/users/dgarcia/data/iu_xray/annotation.json`
- **Images**: `/mnt/sd5/users/dgarcia/data/iu_xray/images`
- **Training batches**: 1034
- **Total training steps**: 15510 (1034 steps × 15 epochs)

---

## Final Metrics (Epoch 14/15)

### Loss
- Final training loss: 1.140

### Evaluation Metrics
- **BLEU-1**: 0.3583444058895111
- **BLEU-2**: 0.22291839122772217
- **BLEU-3**: 0.14960701763629913
- **BLEU-4**: 0.10087913274765015
- **ROUGE-L**: 0.2766414718761074
- **CIDEr**: 0.2446174394031255

---

## Output Artifacts

All outputs saved to:
```
/mnt/sd5/users/dgarcia/R2GenGPT/save/iu_xray/v1_shallow_rep/
```

### Contents
- `log.txt` - Main training log (3.6 MB)
- `logs/csvlog/version_1/metrics.csv` - CSV metrics throughout training
- `logs/csvlog/version_1/hparams.yaml` - Hyperparameters snapshot
- `logs/tensorboard/version_1/` - TensorBoard event files

---

## Environment Reproducibility

To replicate this exact environment:

```bash
# Install from freeze
bash environment_freeze/install_from_freeze.sh /path/to/new/env

# Or use pip directly
python3 -m venv new_env
source new_env/bin/activate
pip install -r environment_freeze/dgarcia_r2gengpt_llama3.2_1b_pip_freeze_20260512.txt
```

### Pip Freeze File
See `dgarcia_r2gengpt_llama3.2_1b_pip_freeze_20260512.txt` for the complete list of 127 packages.

---

## Notes

- METEOR evaluation disabled due to missing Java installation
- Warnings about `pad_token_id` and `attention_mask` during generation (expected)
- Job completed with no fatal errors
- Training reached `max_epochs=15` limit
