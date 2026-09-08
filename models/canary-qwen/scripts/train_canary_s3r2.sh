#!/bin/bash
# S3-R2: dropout-only regularization ablation on UWB-ATCC.
# See research_report/FINAL_RESEARCH_PROGRAM.md Section 7 for design.
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_s3r2
echo "Training complete. Checkpoints at: ~/canary-ft/experiments_s3r2/checkpoints/"
