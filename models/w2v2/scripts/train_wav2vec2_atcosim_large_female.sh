#!/bin/bash
# =============================================================================
# train_wav2vec2_atcosim_large_female.sh
# EXP-007 (research_log): speaker-independent ATCOSIM run, female split.
# Trains on train_female (zf1,zf2,gf1), evaluates on test_female (zf3) — a
# speaker-disjoint split, fixing the leakage documented in ISS-001.
# Run from: ~/w2v2-air-traffic
#
# Mirrors train_wav2vec2_atcosim_large.sh but calls the female-split ablation
# script, which writes to a separate --exp root so it cannot collide with
# the existing (leaked-split) baseline checkpoint or the male run.
# =============================================================================

set -euo pipefail

REPO_DIR="${1:-$HOME/w2v2-air-traffic}"
cd "$REPO_DIR"

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/atcosim/train_w2v2_large-60v-female.sh

echo "Training complete. Results in: experiments/results/speaker_independent_female/"
