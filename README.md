# Pilot-to-ATC Speech Recognition Research

Comparative evaluation of end-to-end (Wav2Vec2) and hybrid (Canary-Qwen-2.5B) ASR models for Air Traffic Control communications, fine-tuned on the UWB-ATCC corpus.

## Models

### 1. Wav2Vec2 Large (End-to-End, CTC)
- **Model**: facebook/wav2vec2-large-960h-lv60-self (317M params)
- **Framework**: HuggingFace Transformers
- **WER**: 14.54% (no LM) / 12.69% (with KenLM 4-gram)
- Based on: [idiap/w2v2-air-traffic](https://github.com/idiap/w2v2-air-traffic)
- See: `models/w2v2/`

### 2. Canary-Qwen-2.5B (Hybrid, SALM)
- **Model**: nvidia/canary-qwen-2.5b (2.5B params, 27.8M trainable via LoRA)
- **Framework**: NVIDIA NeMo (speechlm2)
- **WER**: 23.32% (fine-tuned) / 81.49% (zero-shot)
- See: `models/canary-qwen/`

## Dataset
- **UWB-ATCC**: Air Traffic Control Communications corpus
- University of West Bohemia, Prague Airport recordings
- Train: 11,543 utterances (~10.5 hrs) / Test: 2,886 utterances (~2.6 hrs)
- 80/20 split, seed=1234

## Key Findings
- W2V2 achieves lower WER (14.54%) but trains 100% of 317M parameters
- Canary-Qwen achieves 23.32% WER training only 0.97% of 2.87B parameters
- Both trained for 10,000 steps with matched hyperparameters (lr=5e-4, warmup=1000)
- Zero-shot Canary-Qwen: 81.49% -> Fine-tuned: 23.32% (58% absolute improvement)

## System
- Remote Ubuntu workstation (ET335Lambda)
- 4x NVIDIA RTX 2080 Ti (11GB VRAM each)
- No sudo access, Conda-based environments

## Repository Structure
```
Pilot-to-ATC-Research/
├── README.md
├── models/
│   ├── w2v2/                    # Wav2Vec2 CTC pipeline
│   │   ├── docs/                # Setup, progress docs
│   │   └── scripts/             # Training, eval scripts
│   └── canary-qwen/             # Canary-Qwen SALM pipeline
│       ├── docs/                # Setup, progress, results
│       └── scripts/             # Training config, data conversion, eval
└── shared/
    └── data_info.md             # Dataset documentation
```
