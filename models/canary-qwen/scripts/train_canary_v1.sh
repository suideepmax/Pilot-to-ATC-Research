#!/bin/bash
# v1: Canary-Qwen-2.5B LoRA baseline (0.97% params, q_proj+v_proj, no regularization) on UWB-ATCC
# Result: WER 23.32% (verified via fresh training + inference, 2026-09-07/08)
# Time: ~21 hrs on 4x RTX 2080 Ti (NOT ~5.3 hrs as previously documented — see VAL-010)
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_v1
echo "Training complete. Checkpoints at: ~/canary-ft/experiments/checkpoints/"
