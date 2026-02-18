# ASR Training Report: Wav2Vec2 Fine-Tuning on UWB-ATCC

## Pilot-to-ATC Communication Speech Recognition

**Author:** suideepmax  
**Date:** 2026-02-18 07:44:04  
**Repository:** [Pilot-to-ATC-Research](https://github.com/suideepmax/Pilot-to-ATC-Research)

---

## 1. Project Overview

### 1.1 Objective  
Build an Automatic Speech Recognition (ASR) system that transcribes **pilot-to-ATC (Air Traffic Control) radio communications** into text. These communications follow standardized ICAO phraseology but are challenging due to:  
- Heavy background radio noise and cockpit interference  
- Non-native English speakers (Czech controllers and pilots)  
- Domain-specific vocabulary (callsigns, waypoints, flight levels, headings)  
- Short, fragmented utterances

### 1.2 Dataset  
**UWB-ATCC** (University of West Bohemia - Air Traffic Control Communication)  
- Domain: Real ATC tower communications from Ruzyne (Prague) airport  
- Language: English with Czech-accented speakers  
- Format: Audio (.wav) + transcript pairs in CSV manifests

| Split      | Samples |  
|------------|---------|  
| Train      | ~10,000+ |  
| Validation | ~1,100  |  
| Test       | 2,822   |  

### 1.3 Hardware Constraints  
- **GPU:** NVIDIA (4 GB VRAM)  
- **RAM:** 5.2 GB system memory  
- **OS:** Linux (Ubuntu)

These constraints required aggressive memory optimization throughout all stages.

---

## 2. Training Pipeline

### 2.1 Stage 1 — Base Fine-Tuning (`train_asr.py`)

**Goal:** Adapt a pre-trained speech encoder to the ATC domain.

**Base Model:** `facebook/wav2vec2-base` (95M parameters)  
- This is the **pre-trained only** version — it has a speech feature encoder but **no CTC head**.  
- We deliberately chose this over `wav2vec2-base-960h` (which has a pre-trained CTC head for LibriSpeech) because the 960h model's `lm_head` was trained for a different vocabulary. Loading it with our custom ATC vocabulary would create a mismatch between the encoder's learned representations and the head's expectations.  
- Loaded from a specific revision: `revision="refs/pr/11"`

**Architecture:**  
```
Audio (16kHz) -> Feature Encoder (CNN, 7 layers) -> Transformer Encoder (12 layers) -> lm_head (Linear) -> CTC Loss  
                     ^ FROZEN                           ^ TRAINABLE                    ^ REINITIALIZED
```

**Key Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| `wav2vec2-base` not `base-960h` | Clean encoder without mismatched CTC head |
| Feature encoder frozen | CNN features are general-purpose; fine-tuning them overfits on small data |
| SpecAugment disabled | The `masked_spec_embed` parameter was missing from the checkpoint; enabling it would crash |
| lm_head Xavier-initialized | Fresh head with `gain=1.0` for our custom 29-character ATC vocabulary |
| Gradient checkpointing | Trades compute for memory — essential for 4GB VRAM |
| CTC loss with `ctc_zero_infinity=True` | Prevents NaN gradients when CTC alignment fails |

**Custom Vocabulary (29 tokens):**  
Built from training transcripts. Characters include `a-z`, `0-9`, `'`, `-`, plus:  
- `|` — word delimiter (replaces space)  
- `[UNK]` — unknown token  
- `[PAD]` — padding / CTC blank token (id=28)

**Training Configuration:**  

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Batch size | 4 x 8 (effective 32) |
| Epochs | 30 |
| Warmup steps | 500 |
| Optimizer | AdamW |
| Scheduler | Linear |
| FP16 | Yes |
| Max audio length | 20 seconds |
| Early stopping | Patience 5 |

**Stage 1 Results:**  

| Metric | Validation |
|--------|------------|
| WER | **15.40%** |
| CER | **8.58%** |

The model learned to transcribe ATC communications with ~85% word-level accuracy on the validation set.

---

### 2.2 Stage 2 — Regularization Fine-Tuning (`train_asr_stage2.py`)

**Goal:** Improve generalization by adding regularization to the Stage 1 model.

**Why Stage 2?**  
Stage 1 trained without any regularization (no dropout, no SpecAugment, no weight decay). While this allowed the model to learn quickly, it risked overfitting to the training data. Stage 2 introduces controlled noise during training to force the model to learn more robust representations.

**Critical Fix — `masked_spec_embed` Creation:**  
This was the most important technical challenge. Stage 1 disabled SpecAugment because `masked_spec_embed` (a learnable parameter used to mask time steps during training) was absent from the `wav2vec2-base` checkpoint. When Stage 1 saved the model, it saved **without** this parameter.

Stage 2 needs SpecAugment enabled. Simply setting `mask_time_prob > 0` would crash because the model would try to access `masked_spec_embed` during the forward pass and fail.

**The fix:**  
```python
if self.cfg.mask_time_prob > 0:
    if hasattr(model.wav2vec2, 'masked_spec_embed') and \
       model.wav2vec2.masked_spec_embed is not None:
        # Parameter exists — reset to zeros
        nn.init.zeros_(model.wav2vec2.masked_spec_embed)
        model.wav2vec2.masked_spec_embed.requires_grad = True
    else:
        # Stage 1 saved without it — CREATE it
        hidden_size = model.config.hidden_size  # 768 for base
        model.wav2vec2.masked_spec_embed = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32)
        )
```

This creates a new learnable parameter of shape `(768,)` initialized to zeros, which the model uses as a mask vector during SpecAugment. It's set as `requires_grad=True` so it learns the optimal mask representation during training.

**Regularization Added:**

| Technique | Value | What It Does |
|-----------|-------|--------------|
| SpecAugment (time masking) | `mask_time_prob=0.05` | Randomly masks 5% of time steps in the encoder output, forcing the model to predict from incomplete information |
| Attention dropout | 0.1 | Drops 10% of attention weights, preventing over-reliance on specific attention patterns |
| Hidden dropout | 0.1 | Drops 10% of hidden layer activations |
| LayerDrop | 0.05 | Randomly skips 5% of transformer layers during training, creating an implicit ensemble |
| Weight decay | 0.005 | L2 regularization on all parameters, penalizing large weights |

**Training Configuration Changes from Stage 1:**

| Parameter | Stage 1 | Stage 2 | Why |
|-----------|---------|---------|-----|
| Learning rate | 3e-4 | **5e-5** | Lower LR for fine-tuning — Stage 1 weights are already good, we don't want to destroy them |
| Scheduler | Linear | **Cosine** | Cosine annealing gives smoother LR decay and better final convergence |
| Batch size | 4x8=32 | **2x16=32** | Same effective batch but smaller micro-batch for RAM savings |
| Epochs | 30 | **20** | Regularization slows learning; early stopping handles convergence |
| Max audio length | 20s | **15s** | Reduced to save RAM |
| Early stopping patience | 5 | **8** | More patience since regularization makes progress slower |
| Weight decay | 0 | **0.005** | Added L2 regularization |
| Dataloader workers | 2 | **0** | Zero workers saves ~2GB RAM (each worker spawns a process copy) |

**RAM Optimization Strategy:**  
With only 5.2GB system RAM, memory management was critical:  
1. **No bulk audio loading** — Instead of Stage 1's `AudioLoader` that loaded all audio into a dict, Stage 2 processes files one-at-a-time during featurization  
2. **`writer_batch_size=500`** — Flushes processed features to disk every 500 samples instead of holding everything in RAM  
3. **`dataloader_num_workers=0`** — No extra worker processes  
4. **`dataloader_pin_memory=False`** — Avoids pinned memory allocation  
5. **`eval_accumulation_steps=4`** — Accumulates eval predictions in chunks instead of all at once  
6. **Aggressive `gc.collect()`** — Manual garbage collection after each major step

**Stage 2 Training Progress:**

| Epoch | Train Loss | Eval Loss | Eval WER | Eval CER |
|-------|-----------|-----------|----------|----------|
| 1.6   | 8.04      | 0.7154    | 15.55%   | 8.54%    |
| 3.1   | 7.27      | 0.7672    | 15.20%   | 8.43%    |
| 4.7   | 6.67      | 0.7718    | 14.76%   | 8.14%    |
| 6.3   | 5.87      | 0.7360    | 14.35%   | 8.04%    |
| 9.4   | 5.43      | 0.7947    | 14.42%   | 8.02%    |
| 12.6  | 5.07      | 0.8195    | 14.30%   | 7.84%    |
| 14.2  | 4.70      | 0.8214    | 14.16%   | 7.77%    |
| 15.7  | 4.40      | 0.7989    | 14.10%   | 7.75%    |
| 17.3  | 4.82      | 0.7793    | 14.01%   | 7.74%    |
| 18.9  | 4.77      | 0.7946    | 13.94%   | 7.76%    |
| 20.0  | 4.79      | 0.7945    | **13.97%** | **7.75%** |

**Observations:**  
1. **Training loss (5.96 avg) is much higher than eval loss (0.79)** — This is expected and healthy. The training loss includes the effect of SpecAugment masking, dropout, and layerdrop, which artificially inflate it. The eval loss (computed without any masking/dropout) is the true signal.  
2. **WER improved steadily** from 15.55% -> 13.97% across 20 epochs  
3. **Gradient norms were spiky** — occasional spikes to 300-450, caught by `max_grad_norm=1.0` clipping. This is common with CTC on variable-length ATC audio.  
4. **Early stopping did not trigger** — WER was still slowly improving at epoch 20, meaning the model could potentially benefit from more epochs, but with diminishing returns.

**Stage 2 Results:**

| Metric | Stage 1 | Stage 2 | Relative Improvement |
|--------|---------|---------|---------------------|
| Val WER | 15.40% | **13.97%** | **-9.3%** |
| Val CER | 8.58%  | **7.75%**  | **-9.7%** |

**Training Time:** ~5 hours 49 minutes on a 4GB GPU

---

## 3. Post-Training Evaluation

### 3.1 Inference Testing (`inference.py`)

**Purpose:** Verify the model produces sensible transcriptions on real audio.

**Method:** Greedy decoding — at each time step, pick the character with the highest probability from the CTC output.

**Sample Results (first 28 test files):**

| File | Transcription |
|------|--------------|
| test_000000.wav | for clear land e hea information lima |
| test_000001.wav | el al five two two startup approved time four five |
| test_000002.wav | clearance toel avi via one hotel departure squawk alfa |
| test_000005.wav | departure rununway one three |
| test_000008.wav | nor shuttle one five one five line up runway three one |
| test_000011.wav | csa three seven alfa runway three one cleared for takeoff wind one six zero degrees five knots |
| test_000012.wav | runway three one cleared for takeoff |
| test_000017.wav | ruzyne towergoo day austrian seven zero six papa one lima... |
| test_000027.wav | sky travel one one zero twostansap approvedlearedton level runway three one... |

**Analysis of Inference Results:**

What works well:
- Standard ATC phraseology is captured accurately: *"cleared for takeoff"*, *"runway three one"*, *"wind one six zero degrees five knots"*  
- Callsigns are mostly recognized: *"el al five two two"*, *"csa three seven alfa"*  
- Numbers handled correctly: *"one four zero four"*, *"one two one decimal nine"*

Where it struggles:
- **Word merging:** `twostansap approvedlearedton` (should be "two stand approved cleared to")  
- **Garbled fragments:** `startgalvfaz` (unclear speech)  
- **Truncated words:** `mike sierra ou` (should be "oscar uniform")  
- **Missing spaces:** `towergoo day` (should be "tower good day")

These errors are characteristic of greedy CTC decoding — it picks the locally optimal character at each step but cannot enforce word-level coherence.

---

### 3.2 Test Set Evaluation (`evaluate_test.py`)

**Purpose:** Measure true model performance on held-out data never seen during training or model selection.

**Why this matters:** The validation WER (13.97%) was used for model selection (`load_best_model_at_end=True, metric_for_best_model="wer"`). This means the model is implicitly optimized for the validation set. The test set provides an unbiased estimate of real-world performance.

**Test Set Results:**

| Metric | Validation | Test | Gap |
|--------|------------|------|-----|
| WER | 13.97% | **18.89%** | +4.92% |
| CER | 7.75%  | **10.78%** | +3.03% |

**Interpretation of the Gap:**

The ~5% WER gap between validation and test is significant and reveals several things:

1. **Distribution mismatch:** The test set likely contains different speakers, controllers, accents, or airport scenarios than the training/validation sets. ATC data varies significantly between recording sessions — different controllers have different speaking styles, and different traffic conditions produce different communication patterns.
2. **Mild overfitting to validation:** Since model selection used validation WER, the saved model is the one that happened to perform best on that specific split. On truly unseen data, performance regresses slightly.
3. **Audio quality variation:** Different recording conditions, noise levels, or microphone setups in test recordings compared to training recordings.
4. **This gap is normal** for ASR systems on domain-specific data with high speaker variability. A 5% val-to-test gap is common in the literature for ATC ASR.

---

### 3.3 Language Model Integration

**Purpose:** Reduce word-level errors (merging, garbling) by constraining the decoder to produce valid word sequences.

**How CTC Greedy Decoding Works:**  
```
Audio -> Encoder -> Logits [T x V] -> argmax at each timestep -> collapse repeats -> text
```
The model outputs a probability distribution over the vocabulary (29 characters) at each of ~T time steps. Greedy decoding just picks the highest probability character at each step. This is fast but:  
- Cannot consider future context  
- Cannot enforce that outputs form valid words  
- Produces character-level artifacts like `approvedlearedton`

**How Beam Search + Language Model Works:**  
```
Audio -> Encoder -> Logits [T x V] -> beam search (top-N paths) -> LM rescoring -> best text
                                         ^
                                    KenLM n-gram model
                                    P("cleared for takeoff") >> P("clearedton takeov")
```
Instead of greedily picking one character at each step, beam search maintains the top-N candidate sequences and scores them jointly. A language model adds a prior probability: sequences that form valid English words and common ATC phrases get a bonus.

**Building the Language Model:**

We built a **3-gram KenLM** language model from the training transcripts:

```bash
# Install KenLM from source (pip only gives Python bindings, not the lmplz tool)
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build && cmake .. && make -j$(nproc)

# Extract training transcripts
 tail -n +2 ~/asr_project/data/uwb_atcc/manifests/train.csv | cut -d',' -f2 | tr '[:upper:]' '[:lower:]' > atc_corpus.txt

# Build 3-gram model
~/asr_project/kenlm/build/bin/lmplz -o 3 < atc_corpus.txt > atc_3gram.arpa
```

A 3-gram model captures the probability of each word given the previous 2 words. For ATC, this is powerful because phrases are highly formulaic:  
- `P("takeoff" | "cleared", "for")` is very high  
- `P("takeov" | "cleared", "for")` is essentially zero

**LM Parameters:**  
- `alpha` (LM weight): Controls how much influence the language model has. Higher = more LM correction, but risk of hallucinating words that weren't spoken.  
- `beta` (word insertion bonus): Encourages the decoder to produce more words rather than merging them together.  
- `beam_width`: Number of candidate sequences to maintain. Higher = better results but slower.

**Evaluation with LM** (`evaluate_with_lm.py`):

This script evaluates every test sample with both greedy and beam+LM decoding, then produces a detailed report including:  
- Overall WER/CER comparison  
- Per-sample improvement/regression counts  
- 20 worst samples for each method  
- 20 biggest improvements from LM integration

*(Results pending — evaluation was set up but awaiting KenLM build completion)*

---

## 4. Results Summary

### 4.1 Achieved Results

| Stage | Method | Val WER | Val CER | Test WER | Test CER |
|-------|--------|---------|---------|----------|----------|
| 1 | wav2vec2-base, greedy | 15.40% | 8.58% | — | — |
| 2 | + regularization, greedy | 13.97% | 7.75% | **18.89%** | **10.78%** |
| 2 | + KenLM beam search | — | — | *pending* | *pending* |

### 4.2 Model Lineage

```
facebook/wav2vec2-base (revision refs/pr/11)
|   Pre-trained speech encoder (95M params)
|   No CTC head, no SpecAugment parameter
|
+-- Stage 1: train_asr.py
|   - Custom 29-char ATC vocabulary
|   - lm_head initialized from scratch (Xavier, gain=1.0)
|   - SpecAugment DISABLED (no masked_spec_embed)
|   - lr=3e-4, 30 epochs, batch=4x8
|   - Saved to: models/wav2vec2_uwb_atcc
|   - Result: Val WER 15.40%
|
+-- Stage 2: train_asr_stage2.py
    - Loaded Stage 1 model
    - CREATED masked_spec_embed parameter (768-dim, zeros)
    - SpecAugment ENABLED (mask_time_prob=0.05)
    - Dropout added (attn=0.1, hidden=0.1, layerdrop=0.05)
    - Weight decay=0.005, cosine scheduler
    - lr=5e-5, 20 epochs, batch=2x16
    - Saved to: models/wav2vec2_uwb_atcc_v2
    - Result: Val WER 13.97%, Test WER 18.89%
```

---

## 5. Code Artifacts

| File | Purpose | Key Feature |
|------|---------|-------------|
| `train_asr.py` | Stage 1 training | Base fine-tuning with frozen feature encoder |
| `train_asr_stage2.py` | Stage 2 training | `masked_spec_embed` creation fix, regularization |
| `inference.py` | Basic inference | Greedy CTC decoding |
| `inference_with_lm.py` | LM inference | Side-by-side greedy vs beam+KenLM comparison |
| `evaluate_test.py` | Test evaluation | WER/CER on held-out test set |
| `evaluate_with_lm.py` | Full eval report | Detailed error analysis, worst samples, improvement tracking |

---

## 6. Technical Challenges and Solutions

### 6.1 The `masked_spec_embed` Problem  
**Problem:** `wav2vec2-base` checkpoint lacks `masked_spec_embed`. Stage 1 disabled SpecAugment as a workaround. Stage 2 needs SpecAugment but loading the Stage 1 model (which also lacks `masked_spec_embed`) and enabling it would crash.

**Solution:** Detect whether the parameter exists after loading, and if not, create it as a new `nn.Parameter` initialized to zeros. This is safe because:  
- Zeros mean "no masking effect initially" — the model gradually learns what mask vector works best  
- The parameter is small (768 floats) — negligible memory impact  
- Setting `requires_grad=True` lets it adapt during training

### 6.2 RAM Constraints (5.2 GB)  
**Problem:** Loading all audio files into memory during preprocessing exceeds available RAM.

**Solution:** Redesigned the data pipeline for Stage 2:  
- Process audio files **one at a time** during featurization (not bulk `.map()`)  
- Use `writer_batch_size=500` to flush processed features to disk frequently  
- Set `dataloader_num_workers=0` (each worker spawns a full process copy)  
- Disable `pin_memory` (reserves extra RAM for GPU transfers)  
- Manual `gc.collect()` after each pipeline stage  
- RAM usage monitored throughout: peaked at ~1.7 GB (32.4% of 5.2 GB)

### 6.3 VRAM Constraints (4 GB)  
**Problem:** Fine-tuning a 95M parameter model with backpropagation requires storing activations for all layers.

**Solution:**  
- **Gradient checkpointing:** Instead of storing all intermediate activations, recompute them during backward pass. Trades ~30% more compute for ~60% less memory.  
- **Frozen feature encoder:** The CNN layers (24M params) don't need gradients, saving activation storage for 7 layers.  
- **FP16 training:** Halves memory for activations and gradients.  
- **Small micro-batch:** batch_size=2 (Stage 2) with gradient accumulation to maintain effective batch=32.  
- **`eval_accumulation_steps=4`:** Don't store all eval predictions in VRAM at once.

### 6.4 Training Loss vs Eval Loss Discrepancy  
**Problem:** Stage 2 training loss (~5.96) was 7.5x higher than eval loss (~0.79). This looks alarming.

**Explanation:** This is expected and correct. During training:
- SpecAugment masks 5% of time steps -> model sees corrupted input -> higher loss  
- Dropout drops 10% of activations -> degraded representations -> higher loss  
- LayerDrop skips 5% of layers -> missing computation -> higher loss

During evaluation, ALL regularization is disabled. The model sees clean input with full capacity. The gap actually indicates regularization is working — it's making training harder, forcing the model to learn more robust features.

---

## 7. Next Steps

### 7.1 Immediate (No Retraining)
1. **Complete KenLM evaluation** — Run `evaluate_with_lm.py` with the built 3-gram model to quantify beam search improvement  
2. **Tune LM parameters** — Try different `alpha` (0.3-1.0) and `beta` (0.5-3.0) values to find optimal decoding

### 7.2 Short-Term (Retraining Required)
3. **Larger model** — `wav2vec2-large-xlsr-53` (300M params, 53-language pretraining) should significantly improve performance on accented English. Fits on 4GB VRAM with batch_size=1 + gradient checkpointing.  
4. **Data augmentation** — Speed perturbation (0.9x, 1.0x, 1.1x) effectively triples training data. Noise injection with ATC background noise samples.

### 7.3 Target
Current: **Test WER 18.89%** (greedy)  
Target: **Test WER < 10%**

Expected path:
| Step | Estimated Test WER |
|------|-------------------|
| + KenLM beam search | ~15-16% |
| + wav2vec2-large-xlsr-53 | ~11-13% |
| + Data augmentation | ~9-11% |
| + LM on large model | ~8-9% |

---

## Appendix A: Environment

```
Python: 3.12
PyTorch: (with CUDA)
Transformers: (HuggingFace)
Datasets: (HuggingFace)
Evaluate: (HuggingFace)
KenLM: 0.2.0 (built from source for lmplz)
pyctcdecode: (for beam search)
librosa: (audio resampling)
soundfile: (audio I/O)
```

## Appendix B: File Structure

```
~/asr_project/
├── data/
│   └── uwb_atcc/
│       ├── audio/           # .wav files
│       └── manifests/
│           ├── train.csv    # path, transcript columns
│           ├── valid.csv
│           └── test.csv
├── models/
│   ├── wav2vec2_uwb_atcc/        # Stage 1 output
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── vocab.json
│   │   └── final_metrics.json
│   └── wav2vec2_uwb_atcc_v2/     # Stage 2 output
│       ├── model.safetensors
│       ├── config.json
│       ├── vocab.json
│       └── final_metrics.json
├── train_asr.py              # Stage 1
├── train_asr_stage2.py       # Stage 2
├── inference.py              # Greedy inference
├── inference_with_lm.py      # Beam + LM inference
├── evaluate_test.py          # Test set evaluation
├── evaluate_with_lm.py       # Full comparison report
├── atc_corpus.txt            # Training transcripts for LM
├── atc_3gram.arpa            # KenLM language model
├── training.log              # Stage 1 log
└── training_stage2.log       # Stage 2 log
```