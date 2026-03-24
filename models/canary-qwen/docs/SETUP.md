# Canary-Qwen-2.5B Setup Guide

## Model Overview
- **Model**: nvidia/canary-qwen-2.5b (2.5B parameters)
- **Architecture**: SALM (Speech-Augmented Language Model)
  - Encoder: FastConformer (from nvidia/canary-1b-flash)
  - LLM: Qwen3-1.7B
  - Projection: Linear adapter + LoRA
- **Framework**: NVIDIA NeMo 2.8.0+ (trunk install required)
- **License**: CC-BY-4.0

## Prerequisites
- Miniconda installed (no sudo required)
- CUDA 12.1+ driver (verified: 570.207, CUDA 12.8)
- 4x NVIDIA RTX 2080 Ti (11GB each)

## Environment Setup

### 1. Create conda environment
```bash
conda create -n canary_ft python=3.11 -y
conda activate canary_ft
```

### 2. Install PyTorch 2.6 (FSDP2 support required)
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install NeMo from trunk
The pip-released nemo_toolkit does NOT have the speechlm2 module.
Must install from GitHub trunk:
```bash
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```

### 4. Install additional dependencies
```bash
pip install lhotse sentencepiece transformers datasets librosa soundfile
pip install hydra-core omegaconf pytorch-lightning webdataset braceexpand
pip install editdistance jiwer peft
```

### 5. Verify
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())
import nemo; print('NeMo:', nemo.__version__)
from nemo.collections.speechlm2.models import SALM
print('SALM import: OK')
"
```

## Known Issues
- GPU in error state (e.g., from concurrent training) poisons CUDA context system-wide
- NeMo trunk install may have dependency conflicts — install in a fresh conda env
- Canary requires 16kHz mono audio; UWB-ATCC is 8kHz — resampling needed during data conversion
