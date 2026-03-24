#!/bin/bash
# Initial training run - wav2vec2-base on UWB-ATCC corpus
# Result: WER 79.46% (3000 steps, base model, no LM)
# Run from: ~/w2v2-air-traffic

cd ~/w2v2-air-traffic

conda activate w2v2_asr

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

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
