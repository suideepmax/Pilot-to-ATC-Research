# Canary-Qwen-2.5B Fine-Tuning Progress

## Model
- nvidia/canary-qwen-2.5b (2.87B total parameters)
- Architecture: SALM (FastConformer encoder + Qwen3-1.7B LLM decoder)
- Training: LoRA on LLM + modality adapter (27.8M trainable = 0.97%)
- Framework: NVIDIA NeMo 2.8.0rc0 (speechlm2)

## Environment
- Conda env: canary_ft (Python 3.11)
- PyTorch 2.6.0+cu124
- 4x NVIDIA RTX 2080 Ti (11GB each)
- Strategy: FSDP (ModelParallelStrategy, tensor_parallel=1, data_parallel=4)
- Precision: fp16-true (AdamW eps=1e-4 for stability)

## Data
- UWB-ATCC corpus (same train/test split as W2V2)
- Train: 11,543 utterances (10.54 hrs)
- Test: 2,886 utterances (2.63 hrs)
- Audio: resampled to 16kHz mono (original 8kHz)

## Results

### Zero-Shot Baseline
WER: 81.49% (no fine-tuning)

### Run 1: Original LoRA (lr=5e-4, 10k steps)
WER: 23.32% | val_loss best: 0.678
Hyperparameters: lr=5e-4, warmup=1000, dropout=0.01, no SpecAugment, WD=1e-3

### Run 2: Encoder Unfrozen (838.8M params, 29.2%)
WER: 23.82% | val_loss best: 0.649
Same hyperparameters as Run 1, but FastConformer encoder unfrozen.
Training 30x more parameters did not improve WER.

### Run 3: Lower LR (lr=1e-4, 10k steps)
WER: 32.58% | val_loss best: 0.762
Overfits after step 2500, val_loss rises to 1.109 by step 10k.

### Run 4: Research-optimized (lr=3e-5, r=64, 2500 steps)
Failed: NaN at step 1500 (eps=1e-6 too small for fp16)
Could not evaluate due to r=64 vs pretrained r=128 mismatch.

### Run 5: Research-optimized v2 (lr=3e-5, r=128, 2500 steps)
WER: 60.46% | val_loss: 1.419 (still decreasing)
Too few steps at this LR to converge.

### Run 6: v3 - Original LR + Regularization (best result)
WER: 20.70% | val_loss best: 0.581
Hyperparameters: lr=5e-4, warmup=1000, dropout=0.1, SpecAugment ON, WD=1e-2
Same LR as Run 1, but with SpecAugment, 10x dropout, 10x weight decay.
Broke through the 24% plateau. Config: salm_uwb_atcc_v3.yaml

## Learning Curves (500 test samples)

| Step | v3 (best) | Original | Unfrozen |
|------|-----------|----------|----------|
| 0 | 81.49% | 81.49% | 81.49% |
| 500 | 39.51% | 39.14% | 46.34% |
| 1,000 | 32.68% | 45.02% | 57.37% |
| 2,000 | 27.28% | 30.87% | 26.67% |
| 3,000 | 25.08% | 26.28% | 26.91% |
| 5,000 | 23.00% | 24.77% | 24.85% |
| 7,500 | 23.81% | 24.53% | 24.12% |
| 10,000 | 22.30% | 24.53% | 23.89% |

v3 converges faster than the original at every step after 500, and keeps improving where the original plateaued.

## Key Findings

1. The 24% WER plateau in Runs 1-2 was caused by overfitting, not the frozen decoder. Adding regularization (SpecAugment + dropout + weight decay) cut WER from 23.32% to 20.70%.

2. LoRA (0.97% params) with proper regularization outperforms unfreezing the encoder (29.2% params) without it: 20.70% vs 23.82%.

3. Learning rate must stay high (5e-4) for LoRA on small ATC data. Lower LR (1e-4, 3e-5) converges too slowly or overfits differently.

4. NVIDIA's default SALM config has minimal regularization because it was designed for 234k hours. Fine-tuning on 10 hours requires SpecAugment, dropout=0.1, and weight_decay=1e-2.

5. AdamW eps=1e-4 is required for fp16-true on RTX 2080 Ti. Default 1e-8 or research-suggested 1e-6 causes NaN.

## Known Issues & Fixes
- fp16-true + eps=1e-8 causes NaN: use eps=1e-4
- ModelParallelStrategy rejects 16-mixed: use 16-true
- DDP OOM with 2.5B model: use FSDP
- "Too many open files" crash: ulimit -n 65536 + num_workers=1
- LoRA r=64 checkpoints can't be evaluated with pretrained r=128 model
- NeMo FSDP doesn't log metrics to stdout: extract val_loss from checkpoint messages
