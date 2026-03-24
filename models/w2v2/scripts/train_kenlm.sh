#!/bin/bash
# Train 4-gram KenLM language model on UWB-ATCC training transcripts
# Run from: ~/w2v2-air-traffic
# Result: uwb_atcc_4g.binary

cd ~/w2v2-air-traffic

conda activate w2v2_asr

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$HOME/kenlm/build/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniconda3/pkgs/libboost-1.82.0-h6fcfa73_3/lib:$LD_LIBRARY_PATH

bash src/run_train_kenlm.sh
