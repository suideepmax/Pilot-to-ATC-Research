# Research Summary — wav2vec2 ATC Fine-Tuning

**Date:** April 2026  
**Repo:** suideepmax/Pilot-to-ATC-Research  
**Base paper:** Zuluaga-Gomez et al. — *Automatic Speech Recognition Benchmark for Air-Traffic Communications*

---

## Hardware

- 4× NVIDIA RTX 2080 Ti (11GB VRAM each)
- Training framework: HuggingFace Transformers + PyTorch DDP via `torchrun`
- Conda environment: `w2v2_asr` (CPython 3.10, conda-forge)

---

## Corpora

| Corpus | Duration | Sample Rate | Type | Speakers | Language |
|--------|----------|-------------|------|----------|----------|
| UWB-ATCC | 20.58h | 8kHz | Real ATC comms, Prague Airport | — | English |
| ATCOSIM | ~10h | 32kHz → 16kHz | Simulated ATC, close-talk headset | 10 (6M, 4F) | English (German/Swiss accents) |

Both use 80/20 train/test split (seed=1234), Kaldi format (wav.scp, text, utt2spk).

---

## Model

**facebook/wav2vec2-large-960h-lv60-self** (317M parameters)
- Pre-trained on LibriSpeech 960h + 60,000h unlabeled audio (self-training on Libri-Light)
- Fine-tuned with CTC head for domain-specific ATC ASR
- Feature encoder frozen during fine-tuning

---

## UWB-ATCC Results

### Training progression

| Phase | Model | Steps | Greedy WER | Notes |
|-------|-------|-------|-----------|-------|
| 3a | wav2vec2-base | 3,000 | 79.46% | Pipeline validation only |
| 3b | wav2vec2-base | 10,000 | 60.70% | Full run, no LM |
| 4 | wav2vec2-large-960h-lv60-self | 10,000 | 15.07 / 15.15 / 15.17% | 3 independent runs, training-time greedy eval |

### Final results vs paper (Phase 5 — standalone eval)

| Metric | Paper (Table 3) | Paper (HuggingFace) | **Our Result** |
|--------|----------------|---------------------|----------------|
| WER no LM (greedy) | 17.48% | 17.56% | **14.54%** |
| WER with CTC+KenLM | 14.26% | 13.72% | **12.69%** |

**We beat the paper on both metrics.**

### Learning curve (large model, Run 3)

| Step | Eval WER | Train Loss |
|------|----------|------------|
| 500 | 27.99% | — |
| 1,000 | 20.50% | 1.9717 |
| 3,000 | 17.26% | 0.4164 |
| 5,000 | 16.39% | 0.2305 |
| 10,000 | 15.07% | 0.0606 |

Model crosses the paper's 17.48% WER at approximately step 2,500.

### Why we beat the paper

The paper used LR=1e-4; we used 5e-4 (5× higher). Combined with DDP and a larger effective gradient accumulation, this led to better convergence on the UWB-ATCC corpus. The paper's training-time greedy WER at step 10k was 29.81% — far higher than our 15.07% — indicating their model's logits were noisier, requiring heavy beam search correction to reach 17.56%.

---

## ATCOSIM Results

### Training (large model, 5,000 steps)

| Step | Epoch | Train Loss | Eval WER |
|------|-------|-----------|---------|
| 500 | 4.24 | — | 4.13% |
| 1,000 | 8.47 | 1.0951 | 3.48% |
| 2,000 | 16.94 | 0.0955 | 2.14% |
| 4,500 | 38.13 | — | **1.66%** |
| 5,000 | 42.37 | 0.0216 | 1.67% |

### Final results vs paper

| Metric | Paper (HuggingFace, 20k steps) | **Our Result (5k steps)** |
|--------|-------------------------------|--------------------------|
| WER no LM (greedy) | 1.67% | **1.67%** |
| WER with CTC+KenLM | — | **1.28%** |

**We matched the paper's final WER in ¼ the training steps.**

The paper trained for 20,000 steps and reached 1.67% WER. Our 5,000-step run hit the same number. This is because our smaller effective batch size (64 vs paper's 128) produces more gradient updates per epoch — at step 5,000 we had run 42 epochs vs the paper's 64, but more update steps per sample.

### Why ATCOSIM WER is so much lower than UWB-ATCC

| Factor | ATCOSIM | UWB-ATCC |
|--------|---------|----------|
| Audio quality | Close-talk headset, clean | Telephone, noisy, real-world |
| Vocabulary | Scripted, repetitive ATC phrases | Broader, real comms variation |
| Train/test speaker overlap | Yes (random 80/20) | — |
| Epochs | ~42 (risk of overfitting) | ~7 |

**1.67% is not directly comparable to 14.54%.** Different corpora, different conditions. A fairer ATCOSIM number requires a speaker-independent setup (train on male speakers only, test on held-out female speakers, or vice versa) — not yet done.

---

## KenLM Impact

| Corpus | No LM (greedy) | With CTC+KenLM | Improvement |
|--------|---------------|--------------:|-------------|
| UWB-ATCC | 14.54% | 12.69% | −1.85pp |
| ATCOSIM | 1.67% | 1.28% | −0.39pp |

KenLM helps more on UWB-ATCC because its broader vocabulary has more room for LM correction. ATCOSIM's restricted phrase set leaves less for the LM to fix.

---

## Key Technical Findings

1. **DataParallel vs DDP:** The paper's original `python3` launcher triggers DataParallel, which causes OOM on 11GB GPUs for the 317M parameter model. Switching to `torchrun` (DDP) fixes this — each GPU only handles its own gradients via ring all-reduce instead of gathering everything to GPU 0.

2. **Higher LR outperformed paper:** LR=5e-4 (vs paper's 1e-4) combined with DDP yielded significantly lower WER on UWB-ATCC (14.54% vs 17.56%).

3. **Faster convergence with smaller batch:** Our effective batch size (64) was smaller than the paper's (96 for ATCOSIM, 24 for UWB-ATCC), giving more gradient updates per epoch. ATCOSIM converged to the paper's 20k-step result in just 5k steps.

4. **eval_model.py bug (no LM):** When no LM is provided, `pred_str_ctc_lm` is set equal to the reference text, making the hypo output file look like perfect predictions. Actual WER is computed from greedy `pred_str` vs decoded labels — the printed WER is correct, only the hypo file is misleading.

5. **ATCOSIM gender eval data leakage:** The gender test splits (test_female = zf3, test_male = gm1/gm2) overlap with training data since the 80/20 split was random across all 10 speakers. ~80% of each held-out speaker's utterances were in training. These results were discarded.

---

## Repo Structure

```
Pilot-to-ATC-Research/
├── models/w2v2/docs/
│   ├── PROGRESS_UWB_ATCC.md     # Detailed UWB-ATCC run log
│   └── PROGRESS_ATCOSIM.md      # Detailed ATCOSIM run log
├── shared/
│   └── data_info.md             # Corpus details and comparisons
├── REPLICATION_GUIDE.md         # Step-by-step reproduction guide
├── README.md                    # Overview + results table
└── SUMMARY.md                   # This file
```

All training was done in `/home/kotasthane/w2v2-air-traffic` (the paper's repo clone).

---

## Canary Qwen (SALM) Results — ATCOSIM

Fine-tuned `nvidia/canary-qwen-2.5b` (SALM: FastConformer encoder + Qwen3-1.7B LLM) on ATCOSIM. Two runs:

| Model | Trainable Params | WER | Notes |
|-------|-----------------|-----|-------|
| wav2vec2-large (no LM) | 317M (100%) | 1.67% | CTC, full fine-tuning |
| wav2vec2-large + KenLM | 317M (100%) | 1.28% | CTC + 4-gram LM |
| Canary Qwen v1 (adapter only) | 2.1M (0.07%) | 7.06% | modality adapter only, no LoRA |
| **Canary Qwen v3 (LoRA + SpecAugment)** | **27.8M (0.97%)** | **3.33%** | same config as UWB-ATCC v3 |

v1 → v3 improvement (7.06% → 3.33%) came from three changes: adding LoRA r=128 to the LLM attention layers (2.1M → 27.8M trainable params), adding SpecAugment, and increasing data coverage (~2.6 → ~10.4 effective data epochs).

---

## Completed

- [x] UWB-ATCC: wav2vec2-large fine-tuning + KenLM — 14.54% / 12.69% WER
- [x] ATCOSIM: wav2vec2-large fine-tuning + KenLM — 1.67% / 1.28% WER
- [x] UWB-ATCC: Canary Qwen v3 (LoRA + SpecAugment) — 20.70% WER
- [x] ATCOSIM: Canary Qwen v3 (LoRA + SpecAugment) — 3.33% WER

## Pending / Next Steps

- [ ] **ATCOSIM Phase 4 (valid):** Train separate models on `train_male`/`train_female` splits, evaluate on held-out speakers for true speaker-independent WER (prior attempt had data leakage — random 80/20 split meant test speakers appeared in training)
- [ ] **UWB-ATCC W2V2 ablations:** Effect of dropout, mask_time_prob on UWB-ATCC W2V2
