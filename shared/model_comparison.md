# W2V2 vs Canary-Qwen-2.5B: Complete Comparison

## Final Results

| Model | Params Trained | WER | Training Time |
|---|---|---|---|
| W2V2 Large (no LM) | 317M (100%) | 14.54% | ~8.6 hrs |
| W2V2 Large (with KenLM) | 317M (100%) | 12.69% | ~8.6 hrs |
| Canary-Qwen v3 (LoRA + regularization) | 27.8M (0.97%) | 20.70% | ~5.3 hrs |
| Canary-Qwen (LoRA only) | 27.8M (0.97%) | 23.32% | ~5.3 hrs |
| Canary-Qwen (encoder unfrozen) | 838.8M (32.8%) | 23.82% | ~5.3 hrs |
| Canary-Qwen (zero-shot) | 0 | 81.49% | N/A |

## Hyperparameter Ablation (Canary-Qwen)

| Run | LR | Dropout | SpecAugment | Weight Decay | Steps | WER |
|---|---|---|---|---|---|---|
| v3 (best) | 5e-4 | 0.1 | ON | 1e-2 | 10,000 | 20.70% |
| Original LoRA | 5e-4 | 0.01 | OFF | 1e-3 | 10,000 | 23.32% |
| Encoder unfrozen | 5e-4 | 0.01 | OFF | 1e-3 | 10,000 | 23.82% |
| Lower LR | 1e-4 | 0.01 | OFF | 1e-3 | 10,000 | 32.58% |
| Research-optimized | 3e-5 | 0.1 | ON | 1e-2 | 2,500 | 60.46% |

The v3 run shows that the original 24% WER plateau was caused by overfitting, not the frozen decoder. Adding SpecAugment, dropout (0.1), and weight decay (1e-2) while keeping the original lr=5e-4 broke through the ceiling.

## Learning Curves (WER vs Steps, 500 test samples)

| Step | W2V2 | Canary v3 (reg) | Canary (orig) | Canary (unfrozen) |
|------|------|-----------------|---------------|-------------------|
| 0 | N/A | 81.49% | 81.49% | 81.49% |
| 500 | 26.80% | 39.51% | 39.14% | 46.34% |
| 1,000 | 22.70% | 32.68% | 45.02% | 57.37% |
| 2,000 | 19.13% | 27.28% | 30.87% | 26.67% |
| 3,000 | 17.86% | 25.08% | 26.28% | 26.91% |
| 5,000 | 17.42% | 23.00% | 24.77% | 24.85% |
| 7,500 | 15.92% | 23.81% | 24.53% | 24.12% |
| 10,000 | 15.15% | 22.30% | 24.53% | 23.89% |

The v3 config converges faster at every step after 500, and keeps improving at step 10,000 (22.30%) where the original plateaued at 24.53%.

## Architecture Details

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

| Parameter | W2V2 | Canary-Qwen (v3) |
|---|---|---|
| Steps | 10,000 | 10,000 |
| Learning rate | 5e-4 | 5e-4 |
| Warmup steps | 1,000 | 1,000 |
| Gradient clipping | 1.0 | 1.0 |
| Eval interval | 500 steps | 500 steps |
| GPUs | 4x RTX 2080 Ti | 4x RTX 2080 Ti |

## Observations

1. W2V2 (end-to-end CTC) reaches lower WER (14.54%) than Canary-Qwen (20.70%), despite having 9x fewer total parameters. Full fine-tuning of a CTC model provides stronger domain adaptation than LoRA on a hybrid SALM model.

2. Adding SpecAugment, dropout (0.1), and weight decay (1e-2) to the Canary config reduced WER from 23.32% to 20.70%. The original 24% plateau was not an architectural limit but an overfitting problem. NVIDIA's default config was designed for 234k hours and had minimal regularization.

3. Canary-Qwen shows parameter efficiency: 0.97% of parameters trained achieves better WER (20.70% with regularization) than training 32.8% without it (23.82%).

4. Lowering the learning rate hurts: lr=1e-4 gives 32.58% and lr=3e-5 gives 60.46%. The LoRA adapters need aggressive updates (5e-4) to shift from general English to ATC domain, but also need regularization to avoid memorizing the small training set.

5. W2V2 converges faster at every step and keeps improving through 10k steps (15.15%). Canary v3 also keeps improving at 10k (22.30% on 500 samples, 20.70% on full test), but the gap remains ~6%.

6. The frozen Qwen3-1.7B decoder is trained on general web text, not ATC vocabulary (callsigns, flight levels, runway designators). This domain mismatch likely accounts for the remaining WER gap between Canary (20.70%) and W2V2 (14.54%).
