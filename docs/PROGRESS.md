# Research Progress

## Phase 1 - Environment Setup [DONE]
- Conda env created with CPython 3.10 via conda-forge
- Dependencies installed (pyarrow, fsspec, librosa, soundfile, setuptools pinned)
- uconv Python wrapper created (no sudo needed)

## Phase 2 - Data Preparation [DONE]
- Dataset: UWB-ATCC corpus (Air Traffic Control Communications, Prague Airport)
- Source: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
- Raw data: 20.58h of recordings at 8kHz, .wav + .trs format
- Extracted and organized into audio/ and transcripts/ subfolders
- trs2stm parsing done manually (process substitution workaround for bash limitation)
- Text normalization: CP1250 → UTF8, acronym expansion, number expansion, diacritics removal
- Train/test split (80/20, seed=1234):
  - Train: 11,543 utterances
  - Test: 2,886 utterances

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
- Eval WER: 79.46%
- Runtime: 14h 42min

### Analysis
- High WER expected: only 3k steps, no LM, base model, smaller batch than paper
- Purpose: validate full pipeline end-to-end — confirmed working

## Phase 3b - Base Model Full Run: wav2vec2-base, 10k steps [DONE]
### Model
- Model: facebook/wav2vec2-base (95M parameters)
- Same as 3a but with repo's exact hyperparameters from ablations/uwb_atcc/train_w2v2_base.sh

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
- Eval WER: 60.70% (no LM)
- Eval loss: 2.5709
- Runtime: ~48h

### Analysis
- WER improved from 79.46% → 60.70% with proper step count
- Still higher than paper's ~21% because base model lacks self-training pretraining
- Paper uses wav2vec2-large-960h-lv60-self which was pretrained on 60k hours with self-training

## Phase 4 - Large Model Replication: wav2vec2-large-960h-lv60-self [IN PROGRESS]
### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- This is the **exact model used in the paper** (Table 3, UWB-ATCC results)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training (Libri-Light)
- Self-training makes it significantly more robust than the base model for domain shift

### Why this model
- Wav2Vec 2.0 Large uses a deeper transformer (24 layers vs 12 in base)
- Self-training on 60k hours gives it much stronger generalization to unseen domains like ATC
- The paper reports 20-40% relative WER reduction over hybrid ASR baselines using this model

### Hyperparameters (from ablations/uwb_atcc/train_w2v2_large-60v.sh)
- Steps: 10,000
- Per device batch size: 16
- Gradient accumulation: 2 (effective batch = 128 across 4 GPUs)
- Learning rate: 5e-4
- mask_time_prob: 0.01
- Warmup steps: 1,000
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Expected Results (from paper, Table 3)
- Target WER without LM: ~21%
- Target WER with 4-gram LM: ~17.48%
- Currently training — results to be updated on completion

## Phase 5 - Evaluation with Language Model [PENDING]
- Train 4-gram KenLM on UWB-ATCC training transcripts
- Evaluate fine-tuned models with LM decoding
- Expected WER improvement: ~3-5% absolute

## Phase 4 Update - Large Model Training Issue
- DDP (torchrun) worked and reached step 1300/10000
- Training crashed at step 1300 due to vocab size mismatch (label values > vocab_size: 32)
- Crash corrupted CUDA driver state on GPU 1 — machine reboot required
- Checkpoint saved at step 1000 — will resume from there after reboot
- Resume command: torchrun with --resume_from_checkpoint=checkpoint-1000

## Phase 4 - Large Model Replication: wav2vec2-large-960h-lv60-self [DONE]
### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training
- Fine-tuned on: UWB-ATCC corpus

### Hyperparameters (ablations/uwb_atcc/train_w2v2_large-60v.sh)
- Steps: 10,000
- Per device batch size: 1
- Gradient accumulation: 16 (effective batch = 64 across 4 GPUs)
- Learning rate: 5e-4
- mask_time_prob: 0.01
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)
- DDP via torchrun (DataParallel caused OOM)

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Results
- Eval WER: 15.17% (no LM)
- Eval loss: 0.945
- Train loss: 0.4062
- Runtime: 7h 34min

### Comparison with Paper
| | Paper | Ours |
|---|---|---|
| WER (no LM) | 17.48% | **15.17%** |
| Model | wav2vec2-large-960h-lv60-self | wav2vec2-large-960h-lv60-self |
| Steps | 10,000 | 10,000 |

We beat the paper's reported WER by 2.31% absolute.

## Phase 4 - Large Model Replication: wav2vec2-large-960h-lv60-self [DONE - RUN 2]
### Results (confirmed consistent across runs)
- Eval WER: 15.15% (no LM)
- Eval loss: 0.9479
- Train loss: 0.4076
- Runtime: 8h 35min
- Note: Model was retrained after repo wipe — results consistent with Run 1 (15.17%)

### Comparison with Paper
| | Paper | Ours (Run 1) | Ours (Run 2) |
|---|---|---|---|
| WER (no LM) | 17.48% | 15.17% | **15.15%** |

Both runs beat the paper's reported WER consistently.

## Phase 5 - KenLM Language Model Training [IN PROGRESS]
- Training 4-gram KenLM on UWB-ATCC training transcripts
- Command: `bash src/run_train_kenlm.sh`
- Output: `experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary`
- Expected WER improvement after LM fusion: ~3-5% absolute
