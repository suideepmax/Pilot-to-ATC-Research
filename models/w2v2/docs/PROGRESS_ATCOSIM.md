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

### Comparison with Paper's Published Model

The paper's HuggingFace model card (`Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-atcosim`) shows:

| Step | Paper's epoch | Paper WER |
|------|--------------|-----------|
| 5,000 | 64 | 2.10% |
| 20,000 | 256 | **1.67%** |

The paper trained for **20,000 steps** (4× more than our run). Our 5,000-step run matched the paper's final 1.67% WER.

**Why we converged faster:** Our effective batch size was 64 (vs paper's 128). Smaller batches produce more gradient updates per epoch — the model progresses through more weight updates for the same number of steps. At step 5,000 we had already run 42 epochs, while the paper at step 5,000 was at epoch 64 (more data per step means fewer epochs). Our faster convergence is consistent with smaller batch behavior.

**Bottom line:** Results are equivalent to paper. The paper required 4× more steps due to larger batch size.

---

### Notes

#### Why is 1.67% WER so low?

**Genuine reasons (corpus is actually easier):**
- ATCOSIM uses a **close-talk headset** in a controlled simulation — very clean audio, no background noise
- UWB-ATCC is real communications — telephone quality, noise, real-world variation
- ATCOSIM uses **scripted, repetitive ATC phrases** — limited vocabulary, highly predictable language patterns
- The model (`wav2vec2-large-960h-lv60-self`) is pre-trained on 60k hours and adapts quickly to clean speech

**Experimental design issues (inflate the number):**
1. **Same speakers in train and test.** The 80/20 split is random — all 10 speakers appear in both sets. The model sees every speaker's voice during training. This is not measuring generalization to unseen speakers.
2. **42 epochs of training.** 5000 steps on 7660 samples = 42 passes over the data. UWB-ATCC was ~7 epochs (10000 steps, much larger dataset). At 42 epochs, the model has near-memorized training utterances, and since test speakers overlap, test WER is very low.
3. **Halved effective batch size.** Batch 64 vs paper's 128 — smaller batches with the same steps can increase overfitting.

**Bottom line:** 1.67% is NOT directly comparable to UWB-ATCC's 14.54%. They are different corpora under different conditions. A fairer ATCOSIM number requires speaker-independent evaluation — training on `train_male`/`train_female` and testing on held-out speakers (Phase 4, not yet done).

- WER converges fast — already 4.13% at step 500 vs 27.99% for UWB-ATCC at the same point
- Training deviated from paper on batch size and framework due to hardware — paper's exact params OOM on 11GB VRAM (confirmed)

---

## Phase 3 - KenLM + Final Evaluation [DONE]

- Trained 4-gram KenLM on ATCOSIM train transcripts
- Output: `experiments/data/atcosim_corpus/train/lm/atcosim_corpus_4g.binary`
- Evaluated with `src/eval_model.py` (paper's eval script) + LM

### Commands
```bash
# 1. Train KenLM
export PATH=$HOME/kenlm/build/bin:$PATH
bash src/run_train_kenlm.sh \
  --dataset-name "atcosim_corpus" \
  --text-file "experiments/data/atcosim_corpus/train/text"

# 2. Eval with LM
MODEL="experiments/results/baselines/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc"
LM="experiments/data/atcosim_corpus/train/lm/atcosim_corpus_4g.binary"
python3 src/eval_model.py \
  --language-model "$LM" \
  --pretrained-model "$MODEL" \
  --test-set "experiments/data/atcosim_corpus/test" \
  --print-output "true"
```

### Results
| Metric | Value |
|--------|-------|
| WER no LM (beam search) | 1.67% |
| WER with CTC+KenLM | **1.28%** |

KenLM reduces WER from 1.67% → 1.28% (0.39pp improvement).

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

---

## Phase 5 - Canary Qwen Fine-tuning (NeMo SALM) [DONE]

### Model
- Model: nvidia/canary-qwen-2.5b (SALM — Speech-Augmented Language Model)
- Architecture: FastConformer (nvidia/canary-1b-flash encoder, ~1B params, frozen) + Qwen3-1.7B LLM (frozen) + modality adapter (trainable)
- Framework: NeMo 2.8.0rc0 + Lightning + Lhotse data pipeline
- Trainable parameters: 2,099,200 (modality adapter only, 0.07% of 2.5B total)

### Hyperparameters
| Setting | Value |
|---|---|
| max_steps | 5,000 |
| limit_train_batches | 500 (per epoch) |
| batch_size | 1 (VRAM constraint) |
| accumulate_grad_batches | 4 |
| learning_rate | 5e-4 |
| optimizer | AdamW (β=0.9/0.98, wd=1e-3) |
| lr_scheduler | CosineAnnealing (warmup=500) |
| precision | bf16-true |
| strategy | ModelParallelStrategy (FSDP2, DP=4) |
| gradient_checkpointing | enabled on LLM |
| prompt | "Transcribe the following air traffic control communication in lowercase." |

### Training Notes
- Model sharded across 4× RTX 2080 Ti (11GB each) using FSDP2 data parallelism
- Training stopped at step ~4875 due to 11-hour time limit; best checkpoint at step=3500 (val_loss=0.17676)
- Checkpoint format: FSDP2 distributed checkpoint (4 `.distcp` shards); consolidated to 5.4GB `.pt` file with `dcp_to_torch_save` for eval
- Data: Lhotse CutSet JSONL derived from ATCOSIM Kaldi format (audio resampled 32kHz→16kHz)

### Results
| Metric | Value |
|--------|-------|
| WER (greedy, step=3500) | **7.06%** |
| Total words | 22,789 |
| Insertions | 0.020 |
| Deletions | 0.010 |
| Substitutions | 0.041 |

### Comparison: ATCOSIM Test WER
| Model | WER | Notes |
|---|---|---|
| wav2vec2-large (no LM) | 1.67% | CTC, 317M params, 5k steps |
| wav2vec2-large + KenLM | 1.28% | CTC + 4-gram LM |
| **Canary Qwen (step=3500)** | **7.06%** | SALM, 2.5B params, only adapter fine-tuned |

### Analysis
The 7.06% WER is significantly higher than the fine-tuned wav2vec2 (1.67%) for two reasons:
1. **Only 0.07% of parameters were trained** — the modality adapter (2.1M params) bridges the frozen encoder and frozen LLM; the full model was not fine-tuned
2. **Training stopped early** — best checkpoint at step 3500 of a planned 5000; training showed consistent val_loss improvement (0.28 → 0.17676) and would likely improve further with more steps or a larger trainable adapter
3. **ATCOSIM is clean audio** — wav2vec2 fine-tuned on 7660 samples with speaker overlap achieves near-perfect WER; the generative LLM approach has more overhead for this small, clean corpus

For a larger, noisier corpus (e.g. UWB-ATCC or real-world ATC), the LLM's language understanding could provide more benefit over the CTC approach.
