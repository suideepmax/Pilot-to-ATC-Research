# W2V2 vs Canary-Qwen-2.5B: Side-by-Side Comparison

## Model Summary
| | Wav2Vec2 Large | Canary-Qwen-2.5B |
|---|---|---|
| Architecture | End-to-end CTC | Hybrid SALM (encoder-decoder + LLM) |
| Total params | 317M | 2,870M |
| Trained params | 317M (100%) | 27.8M (0.97%) |
| Pretrained on | LibriSpeech 960h + 60k hrs self-training | 234k hrs diverse English speech |
| Framework | HuggingFace Transformers | NVIDIA NeMo (speechlm2) |
| Fine-tune strategy | Full fine-tune (frozen feature encoder) | LoRA + modality adapter |
| Training precision | fp16 mixed | fp16-true (eps=1e-4) |
| Multi-GPU strategy | DDP | FSDP (ModelParallelStrategy) |

## Hyperparameters (Matched)
| Parameter | W2V2 | Canary-Qwen |
|---|---|---|
| Steps | 10,000 | 10,000 |
| Learning rate | 5e-4 | 5e-4 |
| Warmup steps | 1,000 | 1,000 |
| Gradient clipping | 1.0 | 1.0 |
| Eval interval | 500 steps | 500 steps |
| GPUs | 4x RTX 2080 Ti | 4x RTX 2080 Ti |

## Final Results
| Model | WER (no LM) | WER (with LM) |
|---|---|---|
| W2V2 Large | 14.54% | 12.69% |
| Canary-Qwen (fine-tuned) | 23.32% | N/A |
| Canary-Qwen (zero-shot) | 81.49% | N/A |

## Learning Curve (WER vs Steps)
| Step | W2V2 | Canary-Qwen |
|------|------|-------------|
| 0 (zero-shot) | N/A | 81.49% |
| 500 | 26.80% | 39.14% |
| 1,000 | 22.70% | 45.02% |
| 1,500 | 18.45% | - |
| 2,000 | 19.13% | 30.87% |
| 3,000 | 17.86% | 26.28% |
| 5,000 | 17.42% | 24.77% |
| 7,500 | 15.92% | 24.53% |
| 10,000 | 15.15% | 24.53% |

## Key Observations
1. **W2V2 converges faster**: reaches 26.80% WER at step 500 vs Canary's 39.14%
2. **W2V2 achieves lower final WER**: 15.15% vs 24.53%
3. **Canary-Qwen trains far fewer parameters**: 27.8M (0.97%) vs 317M (100%)
4. **Both models plateau similarly**: W2V2 around step 5000, Canary around step 5000
5. **Canary-Qwen shows larger absolute improvement**: 81.49% → 23.32% (58% drop) vs W2V2's convergence from ~99% → 15.15%
6. **Domain adaptation is harder for hybrid models**: despite being pretrained on 234k hrs (vs W2V2's 60k), Canary-Qwen's hybrid architecture with frozen LLM limits domain adaptation compared to W2V2's full fine-tuning
