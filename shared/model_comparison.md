# W2V2 vs Canary-Qwen-2.5B: Side-by-Side Comparison

## Final Results
| Model | Params Trained | WER |
|---|---|---|
| W2V2 Large (no LM) | 317M (100%) | 14.54% |
| W2V2 Large (with KenLM) | 317M (100%) | 12.69% |
| Canary-Qwen (LoRA only) | 27.8M (0.97%) | 23.32% |
| Canary-Qwen (encoder unfrozen) | 838.8M (32.8%) | 23.82% |
| Canary-Qwen (zero-shot) | 0 | 81.49% |

## Key Finding
Training 0.97% vs 32.8% of Canary-Qwen parameters yields nearly identical WER
(23.32% vs 23.82%), suggesting the performance bottleneck is not in the encoder
but in the frozen LLM decoder. LoRA adaptation alone is sufficient.

## Model Summary
| | Wav2Vec2 Large | Canary-Qwen-2.5B |
|---|---|---|
| Architecture | End-to-end CTC | Hybrid SALM (encoder-decoder + LLM) |
| Total params | 317M | 2,870M |
| Pretrained on | LibriSpeech 960h + 60k hrs | 234k hrs diverse English |
| Framework | HuggingFace Transformers | NVIDIA NeMo (speechlm2) |
| Training precision | fp16 mixed | fp16-true (eps=1e-4) |
| Multi-GPU strategy | DDP | FSDP (ModelParallelStrategy) |

## Matched Hyperparameters
| Parameter | W2V2 | Canary-Qwen |
|---|---|---|
| Steps | 10,000 | 10,000 |
| Learning rate | 5e-4 | 5e-4 |
| Warmup steps | 1,000 | 1,000 |
| Gradient clipping | 1.0 | 1.0 |
| Eval interval | 500 steps | 500 steps |
| GPUs | 4x RTX 2080 Ti | 4x RTX 2080 Ti |

## Learning Curve (WER vs Steps)
| Step | W2V2 | Canary (LoRA) |
|------|------|---------------|
| 0 | N/A | 81.49% |
| 500 | 26.80% | 39.14% |
| 1,000 | 22.70% | 45.02% |
| 2,000 | 19.13% | 30.87% |
| 3,000 | 17.86% | 26.28% |
| 5,000 | 17.42% | 24.77% |
| 7,500 | 15.92% | 24.53% |
| 10,000 | 15.15% | 24.53% |

## Observations
1. W2V2 (end-to-end) converges faster and achieves lower WER than Canary-Qwen (hybrid)
2. W2V2 reaches 26.80% at step 500; Canary needs ~2000 steps to reach similar WER
3. Canary-Qwen plateaus at ~24.5% WER regardless of params trained (0.97% or 32.8%)
4. The frozen LLM decoder appears to be the bottleneck in Canary-Qwen's domain adaptation
5. Despite 234k hrs pretraining (vs 60k for W2V2), Canary-Qwen adapts less effectively to the ATC domain
6. Canary-Qwen shows extreme parameter efficiency: 0.97% params achieves same WER as 32.8%
