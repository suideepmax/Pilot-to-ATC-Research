# W2V2 vs Canary-Qwen-2.5B: Complete Comparison

## Final Results
| Model | Params Trained | WER |
|---|---|---|
| W2V2 Large (no LM) | 317M (100%) | 14.54% |
| W2V2 Large (with KenLM) | 317M (100%) | 12.69% |
| Canary-Qwen (LoRA only) | 27.8M (0.97%) | 23.32% |
| Canary-Qwen (encoder unfrozen) | 838.8M (32.8%) | 23.82% |
| Canary-Qwen (zero-shot) | 0 | 81.49% |

## Learning Curves (WER vs Steps)
| Step | W2V2 | Canary LoRA (0.97%) | Canary Unfrozen (32.8%) |
|------|------|---------------------|-------------------------|
| 0 | N/A | 81.49% | 81.49% |
| 500 | 26.80% | 39.14% | 46.34% |
| 1,000 | 22.70% | 45.02% | 57.37% |
| 2,000 | 19.13% | 30.87% | 26.67% |
| 3,000 | 17.86% | 26.28% | 26.91% |
| 5,000 | 17.42% | 24.77% | 24.85% |
| 7,500 | 15.92% | 24.53% | 24.12% |
| 10,000 | 15.15% | 24.53% | 23.89% |

## Key Findings

### 1. End-to-end (W2V2) adapts better to unseen domains
W2V2 achieves 15.15% WER vs Canary-Qwen's best of 23.32%, despite having 9x fewer
total parameters. Full fine-tuning of a CTC model provides stronger domain adaptation
than LoRA-based adaptation of a hybrid SALM model.

### 2. Canary-Qwen shows extreme parameter efficiency
Training 0.97% of parameters (LoRA only) achieves nearly identical WER to training
32.8% (encoder unfrozen): 23.32% vs 23.82%. This 33x increase in trainable parameters
yields no meaningful improvement, indicating the frozen LLM decoder is the bottleneck.

### 3. Warmup instability is worse with more trainable parameters
The encoder-unfrozen model shows higher WER at steps 500-1000 (46-57%) compared to
LoRA-only (39-45%), suggesting more trainable parameters amplify early training
instability under fp16 precision.

### 4. Both Canary configurations plateau at the same WER (~24%)
Regardless of whether 0.97% or 32.8% of parameters are trained, Canary-Qwen
converges to ~24% WER by step 5000. This ceiling is likely imposed by the frozen
LLM decoder (Qwen3-1.7B), which was not trained on ATC-domain text.

### 5. W2V2 converges faster
W2V2 reaches 26.80% WER at step 500. Canary-Qwen needs ~2000-3000 steps to reach
similar performance, despite starting from a model pretrained on 4x more data.

## Model Architecture Details
| | Wav2Vec2 Large | Canary-Qwen-2.5B |
|---|---|---|
| Architecture | End-to-end CTC | Hybrid SALM (encoder + LLM) |
| Total params | 317M | 2,870M |
| Encoder | Wav2Vec2 (24-layer transformer) | FastConformer (from canary-1b-flash) |
| Decoder | Linear CTC head | Qwen3-1.7B LLM |
| Pretrained on | LibriSpeech 960h + 60k hrs | 234k hrs diverse English |
| Framework | HuggingFace Transformers | NVIDIA NeMo (speechlm2) |
| Fine-tune strategy | Full (frozen feature extractor) | LoRA + optional encoder unfreeze |
| Training precision | fp16 mixed (DDP) | fp16-true (FSDP, eps=1e-4) |

## Matched Hyperparameters
| Parameter | W2V2 | Canary-Qwen |
|---|---|---|
| Steps | 10,000 | 10,000 |
| Learning rate | 5e-4 | 5e-4 |
| Warmup steps | 1,000 | 1,000 |
| Gradient clipping | 1.0 | 1.0 |
| Eval interval | 500 steps | 500 steps |
| GPUs | 4x RTX 2080 Ti | 4x RTX 2080 Ti |
| Dataset | UWB-ATCC (11,543 train / 2,886 test) | Same |
