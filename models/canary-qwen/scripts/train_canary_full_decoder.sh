#!/bin/bash
# Adaptation-Scope Branch 3: full Canary-Qwen decoder fine-tune (no LoRA) on UWB-ATCC.
# See research_report/FINAL_RESEARCH_PROGRAM.md Section 6 for the experimental design.
# NOT YET RUN as of 2026-09-08 -- feasibility confirmed via Spike D (1-step smoke
# test, research_log/VALIDATION.md VAL-009), full run not yet launched.
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_full_decoder
echo "Training complete. Checkpoints at: ~/canary-ft/experiments_full_decoder/checkpoints/"
