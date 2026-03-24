#!/bin/bash
# Full replication run - wav2vec2-large-960h-lv60-self on UWB-ATCC corpus
# Result: WER 14.54% (no LM), 12.69% (with LM) - beats paper
# Run from: ~/w2v2-air-traffic
#
# Prerequisites:
# - src/run_asr_fine_tuning.sh must have torchrun instead of python3
# - gradient_checkpointing must be in correct position
# - batch size must be set to 1 with gradient_acc=16
# See docs/SETUP.md for sed commands to apply these fixes

cd ~/w2v2-air-traffic

conda activate w2v2_asr

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$HOME/kenlm/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/pkgs/libboost-1.82.0-h6fcfa73_3/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh
