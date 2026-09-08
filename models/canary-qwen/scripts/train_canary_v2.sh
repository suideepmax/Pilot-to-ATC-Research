#!/bin/bash
# v2: Canary-Qwen-2.5B encoder unfrozen (838.8M / 29.2% params) on UWB-ATCC
# Result: WER 23.82% (verified via real inference against the HuggingFace-hosted
#   consolidated_model.pt, 2026-09-08 — see VAL-011).
# KNOWN ISSUE: the committed config below (salm_uwb_atcc_v2.yaml) and the
#   training_config.yaml uploaded alongside the HF model are BOTH byte-identical
#   to the v1 (frozen-encoder) config — encoder is NOT actually unfrozen on paper
#   in either file, yet the model measurably differs from v1 (23.82% vs 23.32%).
#   The true hyperparameters that produced this result are not currently
#   recoverable from any config file. Only the WER result is verified; do not
#   trust this config to reproduce it. See ISS-007.
set -e
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_v2
echo "Training complete. Checkpoints at: ~/canary-ft/experiments/checkpoints/"
echo "WARNING: this config is known NOT to reproduce the documented 23.82% result — see header comment."
