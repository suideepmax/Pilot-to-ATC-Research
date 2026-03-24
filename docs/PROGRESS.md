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
- Eval WER: 79.46%
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
- Eval WER: 60.70% (no LM)
- Eval loss: 2.5709
- Runtime: ~48h

### Analysis
- WER improved from 79.46% → 60.70% with proper step count
- Still higher than paper's ~21% because base model lacks self-training pretraining
- Paper uses wav2vec2-large-960h-lv60-self which was pretrained on 60k hours with self-training

## Phase 4 - Large Model Replication: wav2vec2-large-960h-lv60-self [DONE]
### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- This is the **exact model used in the paper** (Table 3, UWB-ATCC results)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training (Libri-Light)
- Architecture: 24 transformer layers (vs 12 in base), self-training gives stronger domain generalization

### Hyperparameters
- Steps: 10,000
- Per device batch size: 1 (reduced from 16 due to 11GB VRAM constraint)
- Gradient accumulation: 16 (effective batch = 64 across 4 GPUs — matches paper)
- Learning rate: 5e-4
- mask_time_prob: 0.01
- Warmup steps: 1,000
- fp16: enabled
- Feature encoder: frozen
- GPUs: 4x NVIDIA RTX 2080 Ti (11GB each)
- Multi-GPU: DDP via torchrun (DataParallel caused OOM due to full model copy per GPU)

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

# 5. Reduce batch size in ablations/uwb_atcc/train_w2v2_large-60v.sh
sed -i 's/per_device_train_batch_size=16/per_device_train_batch_size=1/' ablations/uwb_atcc/train_w2v2_large-60v.sh
sed -i 's/gradient_acc=2/gradient_acc=16/' ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh
```

### Results (confirmed across 2 runs)
- Eval WER: 15.15% (no LM, greedy decoding during training)
- Eval WER: 14.54% (no LM, beam search via eval_model.py — use this for comparison)
- Eval loss: 0.945
- Train loss: 0.407
- Runtime: ~8h on 4x RTX 2080 Ti

## Phase 5 - KenLM Language Model + Evaluation [DONE]
### KenLM Training
- Type: 4-gram language model
- Toolkit: KenLM (built from source at ~/kenlm)
- Trained on: UWB-ATCC training transcripts
- Command: `bash src/run_train_kenlm.sh`
- Output: `experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary`

### KenLM Build Requirements (no sudo)
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
MODEL_FOLDER="experiments/results/baselines/wav2vec2-large-960h-lv60-self/uwb_atcc/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc"
LM_FOLDER="experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary"

python3 src/eval_model.py \
  --pretrained-model "$MODEL_FOLDER" \
  --language-model "$LM_FOLDER" \
  --print-output "true" \
  --test-set "experiments/data/uwb_atcc/test"
```

### Final Results vs Paper
| Metric | Paper | Ours |
|---|---|---|
| WER without LM | 17.48% | **14.54%** |
| WER with CTC+LM | 14.26% | **12.69%** |

We beat the paper on both metrics. Replication complete.
