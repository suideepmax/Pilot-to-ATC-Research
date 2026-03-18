#!/bin/bash
# Full replication run - wav2vec2-base on UWB-ATCC corpus
# Result: WER 60.7% (10000 steps, base model, no LM)
# Run from: ~/w2v2-air-traffic

cd ~/w2v2-air-traffic

conda activate w2v2_asr

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_base.sh
