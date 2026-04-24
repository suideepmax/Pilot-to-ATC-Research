#!/bin/bash
# =============================================================================
# data_prepare_atcosim.sh
# Full data preparation for ATCOSIM corpus (no sudo required)
# Run from: ~/w2v2-air-traffic
#
# Downloads the ISO (~2.5GB), extracts with bsdtar, then runs the repo's
# built-in data_prepare_atcosim_corpus.sh pipeline.
# =============================================================================

set -euo pipefail

ISO_URL="http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso"
DATA_DIR="${1:-$HOME/atcosim_data}"
REPO_DIR="${2:-$HOME/w2v2-air-traffic}"

echo "========================================"
echo " ATCOSIM Data Preparation"
echo " Data dir : $DATA_DIR"
echo " Repo dir : $REPO_DIR"
echo "========================================"

mkdir -p "$DATA_DIR/extracted"
cd "$REPO_DIR"

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

# Step 1 - Download ISO (skip if already present)
if [ ! -f "$DATA_DIR/atcosim.iso" ]; then
  echo "[1/3] Downloading ATCOSIM ISO (~2.5GB)..."
  wget "$ISO_URL" -P "$DATA_DIR/" --progress=dot:giga
else
  echo "[1/3] ISO already present at $DATA_DIR/atcosim.iso, skipping download."
fi

# Step 2 - Extract ISO using bsdtar (no sudo/mount needed)
echo "[2/3] Extracting ISO..."
bsdtar -xf "$DATA_DIR/atcosim.iso" -C "$DATA_DIR/extracted" 2>/dev/null || true
echo "Extracted: $(ls $DATA_DIR/extracted)"

# Step 3 - Run corpus prep pipeline
echo "[3/3] Running data preparation pipeline..."
bash data/databases/atcosim_corpus/data_prepare_atcosim_corpus.sh \
  --DATA "$DATA_DIR/extracted"

echo "========================================"
echo " Data preparation complete!"
echo " Train: $(wc -l < experiments/data/atcosim_corpus/train/text) utterances"
echo " Test : $(wc -l < experiments/data/atcosim_corpus/test/text) utterances"
echo "========================================"
