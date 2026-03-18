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

## Phase 3a - Baseline Run: wav2vec2-base [DONE]
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

### Results
- Train loss: 1.3919
- Eval WER: 79.46%
- Runtime: 14h 42min

### Analysis
- WER of 79.46% is high due to:
  1. Base model has weaker speech representations vs large model
  2. Only 3,000 steps (insufficient for domain adaptation)
  3. No language model (LM) used during decoding
  4. Smaller effective batch size than paper (36 vs 64)
- This run served as a pipeline validation — confirmed full training loop works end to end

## Phase 3b - Full Replication: wav2vec2-base [IN PROGRESS]
### Model
- Model: facebook/wav2vec2-base (95M parameters)
- Same architecture as Phase 3a but with paper's exact hyperparameters

### Hyperparameters (matching repo's ablations/uwb_atcc/train_w2v2_base.sh)
- Steps: 10,000
- Per device batch size: 16
- Gradient accumulation: 2 (effective batch = 32)
- Learning rate: 1e-4
- mask_time_prob: 0.01
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)

### Expected Results (from paper)
- Target WER: ~21% (with LM), ~25-30% (without LM)
- Currently training — results to be updated on completion

## Phase 4 - Large Model Replication [PENDING]
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- Script: ablations/uwb_atcc/train_w2v2_large-60v.sh
- Steps: 10,000
- Expected WER: ~17.48% (with LM), ~21% (without LM)

## Phase 5 - Evaluation with Language Model [PENDING]
- Train 4-gram KenLM on UWB-ATCC training transcripts
- Evaluate fine-tuned models with LM decoding
- Expected WER improvement: ~3-5% absolute

## Phase 3b - Full Replication: wav2vec2-base [DONE]
### Results
- Train completed: 10,000 steps
- Eval WER: 60.70% (no LM)
- Eval loss: 2.5709
- Epoch: 111.11
- Eval samples: 2,885

### Analysis
- WER improved from 79.46% (3k steps) to 60.70% (10k steps)
- Still higher than paper's ~21% WER because:
  1. Base model vs large model (paper uses wav2vec2-large-960h-lv60-self)
  2. No language model used during decoding
  3. Large model has self-training pretraining on 60k hours
- Next: run large model replication to match paper results
