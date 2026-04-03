# Canary-Qwen-2.5B Fine-Tuning Progress

## Model
- **Name**: nvidia/canary-qwen-2.5b (2.5B parameters)
- **Architecture**: SALM (FastConformer encoder + Qwen3-1.7B LLM decoder)
- **Training**: LoRA on LLM + modality adapter (27.8M trainable / 2.87B total = 0.97%)
- **Framework**: NVIDIA NeMo 2.8.0rc0 (speechlm2)

## Environment
- Conda env: canary_ft (Python 3.11)
- PyTorch 2.6.0+cu124
- NeMo 2.8.0rc0 (trunk install)
- 4x NVIDIA RTX 2080 Ti (11GB each)
- Training strategy: FSDP (ModelParallelStrategy, tensor_parallel=1, data_parallel=4)
- Precision: fp16-true (with eps=1e-4 AdamW fix for stability)

## Hyperparameters (matched to W2V2 where applicable)
- Steps: 10,000
- Learning rate: 5e-4 (matched to W2V2 large)
- Warmup: 1,000 steps (matched to W2V2 large)
- Gradient clipping: 1.0 (matched to W2V2 large)
- Gradient accumulation: 4
- Batch size: 2 per GPU
- Optimizer: AdamW (eps=1e-4 for fp16 stability)
- LR scheduler: CosineAnnealing (min_lr=1e-6)
- Eval interval: every 500 steps

## Data
- Dataset: UWB-ATCC (same train/test split as W2V2)
- Train: 11,543 utterances (10.54 hrs)
- Test: 2,886 utterances (2.63 hrs)
- Audio: resampled to 16kHz mono (original 8kHz)
- Format: NeMo JSONL manifests + Lhotse cuts

## Results

### Zero-Shot Baseline (no fine-tuning)
- **WER: 81.49%**

### Fine-Tuned (10k steps)
- **WER: 23.32%**
- val_loss final: 0.805 (best: 0.678)
- Improvement: 58.17% absolute over zero-shot

### Comparison with W2V2
| Model | Params Trained | WER (no LM) | WER (with LM) |
|-------|---------------|-------------|----------------|
| W2V2 large | 317M (100%) | 14.54% | 12.69% |
| Canary-Qwen | 27.8M (0.97%) | 23.32% | N/A |

## Known Issues & Fixes
- fp16-true precision causes NaN with default AdamW eps=1e-8 -> fixed with eps=1e-4
- ModelParallelStrategy does not support 16-mixed -> use 16-true
- DDP OOM with 2.5B model on 11GB GPUs -> use FSDP (ModelParallelStrategy)
- "Too many open files" crash at ~step 407 -> ulimit -n 65536 + num_workers=1
