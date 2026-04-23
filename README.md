# Pilot-to-ATC Speech Recognition Research

Comparing end-to-end (Wav2Vec2) and hybrid (Canary-Qwen-2.5B) ASR models for Air Traffic Control communications, fine-tuned on the UWB-ATCC corpus.

## Results

| Model | Params Trained | WER | Training Time |
|---|---|---|---|
| W2V2 Large (no LM) | 317M (100%) | 14.54% | ~8.6 hrs |
| W2V2 Large (with KenLM) | 317M (100%) | 12.69% | ~8.6 hrs |
| Canary-Qwen v3 (LoRA + reg) | 27.8M (0.97%) | 20.70% | ~5.3 hrs |
| Canary-Qwen (LoRA only) | 27.8M (0.97%) | 23.32% | ~5.3 hrs |
| Canary-Qwen (zero-shot) | 0 | 81.49% | N/A |

All models trained for 10,000 steps with lr=5e-4, warmup=1,000, on 4x RTX 2080 Ti.

## Models

### 1. Wav2Vec2 Large (End-to-End, CTC)
- facebook/wav2vec2-large-960h-lv60-self (317M params)
- HuggingFace Transformers, full fine-tuning
- Based on: [idiap/w2v2-air-traffic](https://github.com/idiap/w2v2-air-traffic)
- See: `models/w2v2/`

### 2. Canary-Qwen-2.5B (Hybrid, SALM)
- nvidia/canary-qwen-2.5b (2.87B params, 27.8M trainable via LoRA)
- NVIDIA NeMo (speechlm2), LoRA fine-tuning
- HuggingFace: [suideepmax/canary-qwen-2.5b-atc-lora](https://huggingface.co/suideepmax/canary-qwen-2.5b-atc-lora)
- See: `models/canary-qwen/`

## Dataset
- UWB-ATCC corpus (Air Traffic Control Communications, Prague Airport)
- Train: 11,543 utterances (~10.5 hrs) / Test: 2,886 utterances (~2.6 hrs)
- 80/20 split, seed=1234

## Key Findings

W2V2 reaches lower WER (14.54%) than Canary-Qwen (20.70%) despite 9x fewer total parameters. Full fine-tuning of a CTC model adapts more effectively to ATC domain than LoRA on a hybrid SALM.

However, Canary-Qwen shows extreme parameter efficiency: 0.97% of parameters trained (27.8M out of 2.87B) gets within 6% absolute WER of a fully fine-tuned 317M model.

Adding regularization (SpecAugment, dropout, weight decay) to Canary-Qwen cut WER from 23.32% to 20.70%. The original 24% plateau was overfitting, not an architectural limit. NVIDIA's default config was calibrated for 234k hours, not 10 hours.

## Repository Structure
Pilot-to-ATC-Research/
├── README.md
├── REPLICATION_GUIDE.md
├── models/
│   ├── w2v2/
│   │   ├── docs/
│   │   └── scripts/
│   └── canary-qwen/
│       ├── docs/
│       └── scripts/
└── shared/
├── data_info.md
└── model_comparison.md

## System
- Ubuntu workstation (ET335Lambda)
- 4x NVIDIA RTX 2080 Ti (11GB VRAM each)
- Conda environments, no sudo access
