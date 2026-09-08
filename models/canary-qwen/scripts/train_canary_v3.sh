#!/bin/bash
# v3: Canary-Qwen-2.5B LoRA + regularization (0.97% params, SpecAugment + dropout=0.1 + weight_decay=1e-2) on UWB-ATCC
# Result: WER 20.70% (verified twice, 2026-09-08: local checkpoint re-evaluated in isolation,
#   and independently against the HuggingFace-hosted consolidated_model.pt — bit-for-bit
#   identical WER both times. See VAL-011.)
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_v3
echo "Training complete. Checkpoints at: ~/canary-ft/experiments/checkpoints/"
