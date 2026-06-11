# Research Progress

## Phase 1 - Environment Setup [DONE]
- Conda env created with CPython 3.10 via conda-forge (default channel installs GraalPy — avoid)
- Dependencies installed with pinned versions (pyarrow, fsspec, librosa, soundfile, setuptools)
- uconv Python wrapper created at `~/bin/uconv` (replaces icu-devtools which requires sudo)
- sox installed via conda-forge

## Phase 2 - Data Preparation [DONE]
- Dataset: UWB-ATCC corpus (Air Traffic Control Communications, Prague Airport)
- Source: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
- Raw data: 20.58h of recordings at 8kHz, .wav + .trs format
- Extracted zip → extracted rar → organized into audio/ and transcripts/ subfolders
- trs2stm parsing done manually via for loop (bash process substitution `<()` doesn't work in this context)
- Text normalization pipeline: CP1250 → UTF8, acronym expansion, number expansion, diacritics removal
- Train/test split (80/20, seed=1234):
  - Train: 11,543 utterances
  - Test: 2,886 utterances
- Script: `scripts/data_prepare_uwb_atcc.sh`

## Phase 3a - Pipeline Validation: wav2vec2-base, 3k steps [DONE]
### Model
- Model: facebook/wav2vec2-base (95M parameters)
- Pretrained on: LibriSpeech 960h
- Fine-tuned on: UWB-ATCC corpus (ATC domain)

### Hyperparameters
- Steps: 3,000
- Per device batch size: 12
- Gradient accumulation: 3 (effective batch = 36)
- Learning rate: 1e-4
- Warmup steps: 1,000
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash src/run_asr_fine_tuning.sh \
  --cmd "none" \
  --model-name-or-path "facebook/wav2vec2-base" \
  --dataset-name "experiments/data/uwb_atcc/train" \
  --eval-dataset-name "experiments/data/uwb_atcc/test" \
  --max-steps 3000 \
  --per-device-train-batch-size 12 \
  --gradient-acc 3 \
  --learning_rate "1e-4" \
  --mask-time-prob "0.0" \
  --overwrite-dir "true" \
  --exp "experiments/results"
```

### Results
- Train loss: 1.3919
- Eval WER: 79.46% (greedy decoding)
- Runtime: 14h 42min

### Analysis
- High WER expected: only 3k steps, no LM, base model, smaller batch than paper
- Purpose: validate full pipeline end-to-end — confirmed working

## Phase 3b - Base Model Full Run: wav2vec2-base, 10k steps [DONE]
### Model
- Model: facebook/wav2vec2-base (95M parameters)
- Same as Phase 3a but using repo's exact hyperparameters from `ablations/uwb_atcc/train_w2v2_base.sh`

### Hyperparameters
- Steps: 10,000
- Per device batch size: 16
- Gradient accumulation: 2 (effective batch = 128 across 4 GPUs)
- Learning rate: 1e-4
- mask_time_prob: 0.01
- Warmup steps: 1,000
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_base.sh
```

### Results
- Eval WER: 60.70% (greedy decoding, no LM)
- Eval loss: 2.5709
- Runtime: ~48h

### Analysis
- WER improved from 79.46% → 60.70% with proper step count
- Still higher than paper's ~21% because base model lacks self-training pretraining
- Paper uses wav2vec2-large-960h-lv60-self pretrained on 60k hours with self-training

## Phase 4 - Large Model Replication: wav2vec2-large-960h-lv60-self [DONE]
### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- Exact model used in the paper (Table 3, UWB-ATCC results)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training (Libri-Light)
- Architecture: 24 transformer layers vs 12 in base, self-training gives stronger domain generalization

### Hyperparameters
- Steps: 10,000
- Per device batch size: 1 (reduced from 16 due to 11GB VRAM constraint)
- Gradient accumulation: 16 (effective batch = 64 across 4 GPUs)
- Learning rate: 5e-4
- mask_time_prob: 0.01
- Warmup steps: 1,000
- fp16: enabled, fp16_full_eval: disabled (prevents CUBLAS crash on RTX 2080 Ti)
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)
- Multi-GPU: DDP via torchrun (DataParallel caused OOM — full model copy per GPU)

### Script Modifications Required
The original repo uses `python3` in `src/run_asr_fine_tuning.sh` which triggers DataParallel (DP). DP copies the full 317M model to every GPU during gradient sync causing OOM on 11GB VRAM. Fix: replace with `torchrun` for DDP.
```bash
# 1. Switch python3 to torchrun
sed -i 's/$cmd python3 src\/run_speech_recognition_ctc.py/$cmd torchrun --nproc_per_node=4 src\/run_speech_recognition_ctc.py/' src/run_asr_fine_tuning.sh

# 2. Remove --gradient_checkpointing (causes CUDA graph crash with DDP on PyTorch 1.13)
sed -i 's/--gradient_checkpointing \\//' src/run_asr_fine_tuning.sh

# 3. Fix blank line caused by sed
sed -i '/^  $/d' src/run_asr_fine_tuning.sh

# 4. Re-add --gradient_checkpointing in correct position
sed -i '/--mask_feature_length=$mask_feature_length \\/a\  --gradient_checkpointing \\' src/run_asr_fine_tuning.sh

# 5. Disable fp16 eval to prevent CUBLAS crash on RTX 2080 Ti
sed -i 's/--fp16 \\/--fp16 \\\n  --fp16_full_eval=False \\/' src/run_asr_fine_tuning.sh

# 6. Reduce batch size in ablations/uwb_atcc/train_w2v2_large-60v.sh
sed -i 's/per_device_train_batch_size=16/per_device_train_batch_size=1/' ablations/uwb_atcc/train_w2v2_large-60v.sh
sed -i 's/gradient_acc=2/gradient_acc=16/' ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Results (confirmed across 3 independent runs)
| Run | Greedy WER | Train Loss | Runtime |
|-----|-----------|------------|---------|
| Run 1 | 15.17% | 0.4062 | 7h 34min |
| Run 2 | 15.15% | 0.4076 | 8h 35min |
| Run 3 | 15.07% | 0.4077 | 7h 15min |
| **Average** | **15.13%** | **0.407** | **~7.8h** |

Note: 15.07% is greedy decoding during training. Beam search eval gives 14.54% (see Phase 5).

### Learning Curve (Run 3)
| Step | Eval WER | Train Loss |
|------|----------|------------|
| 500 | 27.99% | - |
| 1000 | 20.50% | 1.9717 |
| 2000 | 18.98% | 0.6024 |
| 3000 | 17.26% | 0.4164 |
| 5000 | 16.39% | 0.2305 |
| 7000 | 15.80% | 0.1333 |
| 9000 | 15.05% | 0.0760 |
| 10000 | 15.07% | 0.0606 |

Model crosses paper's 17.48% WER baseline at approximately step 2,500.

## Phase 5 - KenLM Language Model + Final Evaluation [DONE]
### KenLM Training
- Type: 4-gram language model
- Toolkit: KenLM (built from source at ~/kenlm)
- Trained on: UWB-ATCC training transcripts
- Command: `bash src/run_train_kenlm.sh`
- Output: `experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary`

### KenLM Build (no sudo)
```bash
conda install -c conda-forge cmake boost-cpp -y
cd ~/kenlm && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
export PATH=$HOME/kenlm/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/pkgs/libboost-1.82.0-h6fcfa73_3/lib:$LD_LIBRARY_PATH
```

### Evaluation Command
```bash
bash models/w2v2/scripts/eval_large_model.sh
```

### Final Results vs Paper (confirmed across 2 eval runs)
| Metric | Paper (Table 3) | Paper (HuggingFace card) | Our Run 1 | Our Run 2 | Average |
|--------|----------------|--------------------------|-----------|-----------|---------|
| WER no LM (beam search) | 17.48% | 17.56% | 14.54% | 14.60% | **14.57%** |
| WER with CTC+LM | 14.26% | 13.72% | 12.69% | 12.82% | **12.76%** |

We beat the paper on both metrics consistently across all runs.

Note: slight discrepancy between Table 3 (17.48%) and HuggingFace model card (17.56%) — likely different text normalization at eval time.

---

### Comparison with Paper's Published Model (HuggingFace)

Model card: `Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc`

**Paper's hyperparameters (from model card — Transformers 4.24.0):**

| Parameter | Paper | Our Run |
|-----------|-------|---------|
| Steps | 10,000 | 10,000 |
| Learning rate | 1e-4 | **5e-4** |
| Effective batch size | **24** | **64** (1×16×4GPUs) |
| Gradient accumulation | 1 (none) | 16 |
| Training method | DataParallel | DDP (torchrun, 4 GPUs) |

Note: the model card shows only `train_batch_size: 24` with no `gradient_accumulation_steps` or `total_train_batch_size` line. The Transformers 4.24.0 card generator only writes those lines when grad_acc > 1 and total ≠ train_batch_size. Their absence confirms grad_acc=1 and effective batch=24.

**Paper's training curve (from model card):**

| Step | Epoch | Train Loss | Eval Loss | Eval WER (greedy) |
|------|-------|-----------|-----------|-------------------|
| 500 | 1.06 | 2.9016 | 0.9995 | — |
| 1000 | 2.12 | 0.9812 | 0.3485 | 28.77% |
| 1500 | 3.18 | 0.7842 | 0.2732 | 78.34% |
| 2500 | 5.31 | 0.6527 | 0.2042 | 60.84% |
| 5000 | 10.62 | 0.6605 | 0.1853 | 45.66% |
| 10000 | 21.23 | 0.7287 | 0.1756 | **29.81%** |

Paper's final greedy WER at step 10,000: 29.81%. Beam search + LM decoding brought it down to 17.56% (no LM beam search) and 13.72% (with LM). Our greedy training WER was ~15.07%, beam search gave 14.54% — a much smaller gap, suggesting our model produces cleaner logits.

**Key observation:** We used a 5× higher learning rate (5e-4 vs 1e-4) and achieved lower WER. The higher LR combined with DDP and larger gradient accumulation appears to have resulted in better optimization for this corpus.
