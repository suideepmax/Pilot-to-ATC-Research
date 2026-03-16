# Research Progress

## Phase 1 - Environment Setup [DONE]
- Conda env created with CPython 3.10 via conda-forge
- Dependencies installed
- uconv Python wrapper created

## Phase 2 - Data Preparation [DONE]
- UWB-ATCC corpus downloaded and extracted
- trs2stm parsing done manually (process substitution workaround)
- Train/test split complete:
  - Train: 11,543 utterances
  - Test: 2,886 utterances

## Phase 3 - Training [IN PROGRESS]

## Phase 4 - Evaluation [PENDING]

## Phase 3 - Training [DONE]
- Model: facebook/wav2vec2-base
- Dataset: UWB-ATCC (11,522 train / 2,885 eval utterances)
- Steps: 3,000
- Train loss: 1.3919
- Eval WER: 79.46% (no LM, base model, 3k steps only)
- Checkpoint: experiments/results/wav2vec2-base/uwb_atcc/...
- Runtime: 14h 42min on 4x RTX 2080 Ti

## Phase 4 - Evaluation [IN PROGRESS]
