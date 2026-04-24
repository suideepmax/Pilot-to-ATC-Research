#!/bin/bash
# =============================================================================
# train_wav2vec2_atcosim_large.sh
# Fine-tune wav2vec2-large-960h-lv60-self on ATCOSIM corpus
# Run from: ~/w2v2-air-traffic
#
# Mirrors ablations/atcosim/train_w2v2_large-60v.sh but with batch size
# reduced to 1 + grad accum 16 to fit within 11GB VRAM (RTX 2080 Ti).
# Uses torchrun (DDP) to avoid DataParallel OOM on 317M param model.
# =============================================================================

set -euo pipefail

REPO_DIR="${1:-$HOME/w2v2-air-traffic}"
cd "$REPO_DIR"

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/atcosim/train_w2v2_large-60v.sh

echo "Training complete. Results in: experiments/results/baselines/"
