#!/bin/bash
# Evaluate wav2vec2-large-960h-lv60-self with and without KenLM
# Run from: ~/w2v2-air-traffic

cd ~/w2v2-air-traffic

conda activate w2v2_asr

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$HOME/kenlm/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/pkgs/libboost-1.82.0-h6fcfa73_3/lib:$LD_LIBRARY_PATH

MODEL_FOLDER="experiments/results/baselines/wav2vec2-large-960h-lv60-self/uwb_atcc/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc"
LM_FOLDER="experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary"

python3 src/eval_model.py \
  --pretrained-model "$MODEL_FOLDER" \
  --language-model "$LM_FOLDER" \
  --print-output "true" \
  --test-set "experiments/data/uwb_atcc/test"
