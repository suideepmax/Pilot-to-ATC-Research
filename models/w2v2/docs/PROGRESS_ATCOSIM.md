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

| Setting | Paper (original) | This run | Note |
|---|---|---|---|
| max_steps | 5,000 | 5,000 | ✓ same |
| learning_rate | 5e-4 | 5e-4 | ✓ same |
| mask_time_prob | 0.01 | 0.01 | ✓ same |
| per_device_train_batch_size | 16 | 1 | ✗ VRAM constraint (RTX 2080 Ti, 11GB) |
| gradient_accumulation | 2 | 16 | ✗ compensates batch, but effective batch = 64 vs paper's 128 |
| training method | python3 (DataParallel) | torchrun (DDP) | ✗ OOM with DP on 317M model |
| min_duration_in_seconds | 0.2 | 0.5 | ✗ 45 ATCOSIM clips crash mask (seq_len < mask_len=12) |
| fp16_full_eval | not set | False | ✗ prevents CUBLAS crash on RTX 2080 Ti |
| gradient_checkpointing | yes | yes | ✓ same |

**Training was NOT identical to the paper.** Effective batch size was halved (64 vs 128) and DDP was used instead of DataParallel due to hardware constraints.

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

### Eval (paper's eval_model.py, no LM)
Run command:
```bash
python3 src/eval_model.py \
  --pretrained-model $MODEL \
  --test-set experiments/data/atcosim_corpus/test \
  --print-output true
```
Result: **WER 1.67%** — matches step-5000 training eval exactly.

### Notes
- WER converges fast — already 4.13% at step 500 vs 27.99% for UWB-ATCC at the same point
- Model nearly plateaus after step 2000 (~2.1%), then slowly improves to 1.67%
- ATCOSIM is a much easier corpus than UWB-ATCC: close-talk headset, controlled simulation environment, no background noise
- Training deviated from paper on batch size and training framework due to hardware limits — results are not directly comparable to paper's reported numbers

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

### Status: NOT a valid speaker-independent eval

The gender splits (test_female = zf3, test_male = gm1/gm2) are subsets of the **full** corpus filtered by speaker. The model was trained on a **random 80/20 split of all 10 speakers**, so ~80% of zf3's and gm1/gm2's utterances were already in the training set. This is data leakage — not a true unseen-speaker evaluation.

To do a valid speaker-independent eval, a separate model would need to be trained on `train_female` / `train_male` splits and tested on the corresponding held-out speakers. That is a different experiment (Phase 4 remains TODO).

### Discarded numbers (not trustworthy due to data leakage)
| Split | eval_model.py WER | Custom script WER | Valid? |
|---|---|---|---|
| test_female (zf3) | 86% | 0.86% | ✗ No — ~80% of utterances were in training |
| test_male (gm1, gm2) | 1.23% | 0.01% | ✗ No — same issue |

Note: eval_model.py also has a bug (uses CTC-decoded labels as reference) that inflates WER vs the custom script.
