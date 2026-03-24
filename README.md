# Pilot-to-ATC Speech Recognition Research

Fine-tuning speech recognition models for Air Traffic Control (ATC) communications using the UWB-ATCC corpus.

## Models

### 1. Wav2Vec2 (CTC-based)
- Based on: [idiap/w2v2-air-traffic](https://github.com/idiap/w2v2-air-traffic)
- Framework: HuggingFace Transformers
- Architecture: Wav2Vec2 encoder + CTC decoder
- Status: Baseline trained (WER 60.70% at 10k steps, base model, no LM)
- See: `models/w2v2/`

### 2. Canary-Qwen-2.5B (SALM-based)
- Based on: [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b)
- Framework: NVIDIA NeMo (speechlm2)
- Architecture: FastConformer encoder + Qwen3-1.7B LLM decoder (SALM)
- Status: Environment setup complete, data conversion in progress
- See: `models/canary-qwen/`

## Dataset
- **UWB-ATCC**: Air Traffic Control Communications corpus
- Source: University of West Bohemia, Prague Airport recordings
- ~20.58 hours, 8kHz, Czech ATC in English
- Train: 11,543 utterances / Test: 2,886 utterances

## System
- Remote Ubuntu workstation (ET335Lambda)
- 4x NVIDIA RTX 2080 Ti (11GB VRAM each)
- No sudo/admin access
- Conda-based environment management

## Repository Structure
```
Pilot-to-ATC-Research/
├── README.md
├── models/
│   ├── w2v2/                    # Wav2Vec2 CTC pipeline
│   │   ├── docs/                # Setup, progress docs
│   │   └── scripts/             # Training, eval scripts
│   └── canary-qwen/             # Canary-Qwen SALM pipeline
│       ├── docs/                # Setup, progress docs
│       └── scripts/             # Training, data conversion, eval scripts
└── shared/
    └── data_info.md             # Dataset documentation
```
