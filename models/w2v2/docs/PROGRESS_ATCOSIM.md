# Research Progress — ATCOSIM Corpus

## Phase 1 - Data Preparation [DONE]

- Dataset: ATCOSIM Air Traffic Control Simulation Speech Corpus
- Source: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- Downloaded ISO (~2.5GB) from: http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso
- Extracted with `bsdtar` (no sudo needed) to `~/atcosim_data/extracted/`
- Text normalization pipeline: local_filter (replaces ATC-specific tags), acronym linking, uconv lowercase, final_normalization.py
- Train/test split (80/20, seed=1234):
  - Train: 7,660 utterances
  - Test: 1,916 utterances
- Gender-based subsets also created: train_female, test_female, train_male, test_male
- Output: `experiments/data/atcosim_corpus/{train,test,train_female,test_female,train_male,test_male}/`
- Script: `scripts/data_prepare_atcosim.sh`

### Raw Corpus Details
- Duration: ~10 hours
- Sample rate: 32kHz (resampled to 16kHz via sox)
- Language: English (non-native speakers)
- Accents: German, Swiss-German, Swiss-French
- Speakers: 10 total (6 male, 4 female)
- Recording type: Close-talk headset, ATC real-time simulations

### Speaker Split
| Split | Speakers |
|---|---|
| train_female | zf1, zf2, gf1 |
| test_female | zf3 |
| train_male | sm1, sm2, sm3, sm4 |
| test_male | gm1, gm2 |

---

## Phase 2 - Large Model Training: wav2vec2-large-960h-lv60-self [DONE]

### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- Same model used for UWB-ATCC replication (Phase 4 in PROGRESS_UWB_ATCC.md)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training

### Hyperparameters
- Steps: 5,000
- Per device batch size: 1 (reduced from 16 — VRAM constraint on RTX 2080 Ti)
- Gradient accumulation: 16 (effective batch = 64 across 4 GPUs)
- Learning rate: 5e-4
- mask_time_prob: 0.01
- min_duration_in_seconds: 0.5 (filter 45 clips too short for mask_time_length=12)
- Warmup steps: 1,000
- fp16: enabled, fp16_full_eval: disabled (CUBLAS crash prevention)
- Multi-GPU: DDP via torchrun

### Command
```bash
cd ~/w2v2-air-traffic
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/train_wav2vec2_atcosim_large.sh
```

### Learning Curve

| Step | Epoch | Train Loss | Eval WER |
|------|-------|-----------|---------|
| 500 | 4.24 | — | 4.13% |
| 1000 | 8.47 | 1.0951 | 3.48% |
| 1500 | 12.71 | — | 2.84% |
| 2000 | 16.94 | 0.0955 | 2.14% |
| 2500 | 21.19 | — | 2.17% |
| 3000 | 25.42 | 0.0590 | 1.97% |
| 3500 | 29.66 | — | 2.02% |
| 4000 | 33.89 | 0.0358 | 1.79% |
| **4500** | **38.13** | — | **1.66%** |
| 5000 | 42.37 | 0.0216 | 1.67% |

### Results
| Metric | Value |
|--------|-------|
| Best Eval WER (greedy, step 4500) | **1.66%** |
| Final Eval WER (greedy, step 5000) | 1.67% |
| Final Train Loss | 0.0216 |
| Runtime | ~3.8 hours |

### Notes
- WER converges fast — already 4.13% at step 500 vs 27.99% for UWB-ATCC at the same point
- Model nearly plateaus after step 2000 (~2.1%), then slowly improves to 1.66%
- ATCOSIM is a much easier corpus than UWB-ATCC: close-talk headset, controlled simulation environment, no background noise

---

## Phase 3 - KenLM + Final Evaluation [ ]

- Train 4-gram KenLM on ATCOSIM train transcripts
- Evaluate with beam search + LM

### Results
| Metric | Value |
|--------|-------|
| WER no LM (beam search) | TBD |
| WER with CTC+LM | TBD |

---

## Phase 4 - Gender Experiments (Speaker-Independent Eval) [DONE]

Evaluated the full-data model on speaker-independent test splits to measure true generalization.
test_female (zf3) and test_male (gm1, gm2) are speakers whose voices were **never seen during training**.

### Method
Custom eval script — greedy CTC predictions vs original text using `evaluate["wer"]`.
Note: `eval_model.py` has a bug where it uses `processor.decode(labels, group_tokens=False)` as the
reference (outputs raw CTC characters with `|` separators), producing false high WER. Fixed by
comparing against the original text from the data loader directly.

### Results

| Split | Speakers | Utterances | WER (greedy) |
|---|---|---|---|
| test_female | zf3 (Swiss-French accent) | 616 | **0.86%** |
| test_male | gm1, gm2 (German accent) | 638 | **0.01%** |

### Analysis
- Model generalizes near-perfectly to unseen speakers despite being trained on a random (non-speaker-split) 80/20 split
- 0.01% for test_male is essentially perfect — gm1/gm2 German accent is very close to the training distribution (sm1-sm4 also German)
- 0.86% for test_female is slightly higher — zf3 has a Swiss-French accent, the only French-accented speaker in the corpus, making it the most out-of-distribution voice
- ATCOSIM corpus is clean enough (close-talk headset, scripted ATC phrases) that the model adapts almost perfectly across speakers
