# Canary-Qwen-2.5B Fine-Tuning Progress

## Model
- **Corrected 2026-09-10 (see research_log/ISSUES.md ISS-013)**: this is NOT the released
  `nvidia/canary-qwen-2.5b` checkpoint fine-tuned in place. Every config in this project
  (v1/v2/v3 and the newer S3-B3 track) builds a SALM by composing `pretrained_llm:
  Qwen/Qwen3-1.7B` + `pretrained_asr: nvidia/canary-1b-flash` via NeMo's speechlm2
  recipe, with `perception.proj` (the modality-adapter projection connecting the two)
  left at random initialization -- `canary-1b-flash`'s own checkpoint has no such layer,
  and this project's training scripts never load the released `canary-qwen-2.5b`
  checkpoint (which does contain a trained projection) to initialize it.
- Architecturally the same design as `nvidia/canary-qwen-2.5b` (FastConformer encoder +
  Qwen3-1.7B LLM decoder, SALM composition), but a separately-assembled instance with
  its own (randomly-initialized, then trained) modality projection -- not the released
  model's weights for that component.
- Training: LoRA on LLM + modality adapter (27.8M trainable = 0.97%)
- Framework: NVIDIA NeMo 2.8.0rc0 (speechlm2)
- **Evaluation provenance not yet audited** for this distinction (ISS-013, OPEN) --
  do not treat any WER number below as verified-correct-checkpoint until that audit
  completes.

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

Naming convention (as of 2026-09-08): the three canonical, actively-maintained runs are **v1** (LoRA baseline), **v2** (encoder unfrozen), **v3** (LoRA + regularization, best result) — matching `salm_uwb_atcc_v1/v2/v3.yaml` and `train_canary_v1/v2/v3.sh`. All three WER numbers below were independently re-verified via real inference on 2026-09-08 (not just cited from earlier records) — see `research_log/VALIDATION.md` VAL-010/VAL-011.

### Zero-Shot Baseline
WER: 81.49% (no fine-tuning)

### v1: LoRA baseline (lr=5e-4, 10k steps)
WER: 23.32% | val_loss best: 0.678
Hyperparameters: lr=5e-4, warmup=1000, dropout=0.01, no SpecAugment, WD=1e-3
Verified 2026-09-08 via a fresh training run (not just the historical record) — reproduced to within rounding. **Actual training time: ~21 hours** on 4x RTX 2080 Ti (not ~5.3 hours as earlier documented — that figure was wrong).

### v2: Encoder Unfrozen (838.8M params, 29.2%)
WER: 23.82% | val_loss best: 0.649
Documented hyperparameters: same as v1 but FastConformer encoder unfrozen. **However, both the committed config and the config uploaded alongside the published HuggingFace model are byte-identical to v1's config (encoder still shown frozen on paper) — the true hyperparameters that produced this measurable, reproducible 23.82% result are not currently recoverable from any file.** Only the WER number is verified (via real inference against the published model); the config should not be trusted to reproduce it. See `research_log/ISSUES.md` ISS-007.
Training 29.2% of parameters did not improve WER over v1's 0.97%.

### Lower LR (lr=1e-4, 10k steps) — checkpoint lost, not part of the v1/v2/v3 set
WER: 32.58% (historical citation only, not independently re-verified)
Overfits after step 2500, val_loss rises to 1.109 by step 10k.
No surviving checkpoint anywhere (local disk or HuggingFace) — config survives locally as `~/canary-ft/conf/salm_uwb_atcc_lr1e4.yaml` (not committed to GitHub). **Planned for a fresh, lower-priority retrain** — see `research_log/EXPERIMENTS.md`.

### Research-optimized ablations — removed from the active record (2026-09-08)
Two lr=3e-5 variants were previously attempted (r=64/4-projection LoRA, which NaN'd at step 1500 and was never evaluable; and r=128/2-projection, which reached 60.46% WER but never converged in the 2,500-step budget). Per user decision, these are dropped from the manuscript and active research record — neither has a surviving checkpoint, and they are not considered part of the core v1/v2/v3 comparison going forward.

### v3: LoRA + Regularization (best result)
WER: 20.70% | val_loss best: 0.581
Hyperparameters: lr=5e-4, warmup=1000, dropout=0.1, SpecAugment ON, WD=1e-2
Same LR as v1, but with SpecAugment, 10x dropout, 10x weight decay.
Broke through the 24% plateau. Config: `salm_uwb_atcc_v3.yaml`.
**Verified twice on 2026-09-08**: the surviving local checkpoint (re-evaluated in isolation after an earlier evaluation was contaminated by a caching bug) and an independently-downloaded copy from the published HuggingFace model both gave bit-for-bit identical WER (0.2070041846744872) — this result is genuine and reproducible.

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

v3 converges faster than v1 at every step after 500, and keeps improving where v1 plateaued.

## Key Findings

1. The 24% WER plateau in v1/v2 was caused by overfitting, not the frozen decoder. Adding regularization (SpecAugment + dropout + weight decay) cut WER from 23.32% to 20.70%.

2. LoRA (0.97% params) with proper regularization outperforms unfreezing the encoder (29.2% params) without it: 20.70% vs 23.82%.

3. Learning rate must stay high (5e-4) for LoRA on small ATC data. Lower LR (1e-4) converges too slowly or overfits differently (citation only — not independently re-verified, checkpoint lost).

4. NVIDIA's default SALM config has minimal regularization because it was designed for 234k hours. Fine-tuning on 10 hours requires SpecAugment, dropout=0.1, and weight_decay=1e-2.

5. AdamW eps=1e-4 is required for fp16-true on RTX 2080 Ti. Default 1e-8 or research-suggested 1e-6 causes NaN.

## Known Issues & Fixes
- fp16-true + eps=1e-8 causes NaN: use eps=1e-4
- ModelParallelStrategy rejects 16-mixed: use 16-true
- DDP OOM with 2.5B model: use FSDP
- "Too many open files" crash: ulimit -n 65536 + num_workers=1
- LoRA r=64 checkpoints can't be evaluated with pretrained r=128 model
- NeMo FSDP doesn't log metrics to stdout: extract val_loss from checkpoint messages
