# Research Progress — ATCOSIM Corpus

## Phase 1 - Data Preparation [DONE]

- Dataset: ATCOSIM Air Traffic Control Simulation Speech Corpus
- Source: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- Downloaded ISO (~2.5GB) from: http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso
- Extracted with `bsdtar` (no sudo needed) to `~/atcosim_data/extracted/`
- Text normalization pipeline: local_filter (replaces ATC-specific tags), acronym linking, uconv lowercase, final_normalization.py
- Train/test split (80/20, seed=1234):
  - Train: TBD utterances
  - Test: TBD utterances
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

## Phase 2 - Large Model Training: wav2vec2-large-960h-lv60-self [ ]

### Model
- Model: facebook/wav2vec2-large-960h-lv60-self (317M parameters)
- Same model used for UWB-ATCC replication (Phase 4 in PROGRESS_UWB_ATCC.md)
- Pretrained on: LibriSpeech 960h + 60,000h unlabeled audio via self-training

### Planned Hyperparameters (from `ablations/atcosim/train_w2v2_large-60v.sh`)
- Steps: 5,000
- Per device batch size: 16 → **reduce to 1** (VRAM constraint on RTX 2080 Ti, same fix as UWB-ATCC)
- Gradient accumulation: 2 → **increase to 16** (maintain effective batch = 64)
- Learning rate: 5e-4
- mask_time_prob: 0.01
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

### Results
| Metric | Value |
|--------|-------|
| Eval WER (greedy) | TBD |
| Train loss | TBD |
| Runtime | TBD |

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

## Phase 4 - Gender Experiments [ ]

Train and evaluate on gender-split subsets to measure WER variation by speaker accent/gender.

| Split | WER |
|---|---|
| train_female / test_female | TBD |
| train_male / test_male | TBD |
| full train / test_female | TBD |
| full train / test_male | TBD |
