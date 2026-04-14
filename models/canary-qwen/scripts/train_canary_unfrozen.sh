#!/bin/bash
# Train Canary-Qwen-2.5B encoder unfrozen (32.8% params) on UWB-ATCC
# Result: WER 23.82% | Time: ~5.3 hrs on 4x RTX 2080 Ti
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_unfrozen
echo "Training complete. Checkpoints at: ~/canary-ft/experiments/checkpoints/"
