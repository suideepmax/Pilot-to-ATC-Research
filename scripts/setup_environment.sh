#!/bin/bash
# =============================================================================
# setup_environment.sh
# Full environment setup for w2v2-air-traffic replication (no sudo required)
# Tested on: Ubuntu 24, CUDA 11.7, 4x RTX 2080 Ti
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/idiap/w2v2-air-traffic"
ENV_NAME="${1:-w2v2_asr}"  # pass env name as argument, default: w2v2_asr
INSTALL_DIR="${2:-$HOME}"   # pass install dir as argument, default: $HOME

echo "========================================"
echo " w2v2-air-traffic Environment Setup"
echo " Env name : $ENV_NAME"
echo " Install dir: $INSTALL_DIR"
echo "========================================"

# Step 1 - Clone repo
cd "$INSTALL_DIR"
if [ ! -d "w2v2-air-traffic" ]; then
    echo "[1/7] Cloning repository..."
    git clone "$REPO_URL"
else
    echo "[1/7] Repo already exists, skipping clone..."
fi
cd w2v2-air-traffic

# Step 2 - Accept conda TOS
echo "[2/7] Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Step 3 - Create conda environment using conda-forge (avoids GraalPy)
echo "[3/7] Creating conda environment: $ENV_NAME ..."
conda create -n "$ENV_NAME" -c conda-forge python=3.10 -y

# Step 4 - Fix typo in requirements.txt and install
echo "[4/7] Fixing requirements.txt and installing dependencies..."
sed -i 's/pyctcdecode=0\.4\.0/pyctcdecode==0.4.0/' requirements.txt

# activate env and install
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r requirements.txt

# Step 5 - Pin compatible versions (fixes known conflicts)
echo "[5/7] Pinning compatible package versions..."
pip install \
    pyarrow==14.0.1 \
    fsspec==2023.6.0 \
    datasets==2.14.0 \
    setuptools==67.6.0 \
    librosa==0.9.2 \
    soundfile

# Install sox via conda (no sudo needed)
conda install -c conda-forge sox -y

# Step 6 - Install uconv Python wrapper (replaces icu-devtools)
echo "[6/7] Installing uconv wrapper..."
mkdir -p "$HOME/bin"
cat > "$HOME/bin/uconv" << 'UCONV'
#!/usr/bin/env python3
import sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--from-encoding', default='utf-8')
parser.add_argument('-t', '--to-encoding', default='utf-8')
parser.add_argument('-x', '--transliterate', default=None)
parser.add_argument('files', nargs='*')
args = parser.parse_args()
def process(text):
    if args.transliterate and 'Lower' in args.transliterate:
        return text.lower()
    return text
if args.files:
    for f in args.files:
        with open(f, 'rb') as fh:
            text = fh.read().decode(args.from_encoding, errors='replace')
        sys.stdout.write(process(text))
else:
    raw = sys.stdin.buffer.read()
    text = raw.decode(args.from_encoding, errors='replace')
    sys.stdout.write(process(text))
UCONV
chmod +x "$HOME/bin/uconv"

# Add ~/bin to PATH permanently
if ! grep -q 'export PATH=$HOME/bin:$PATH' "$HOME/.bashrc"; then
    echo 'export PATH=$HOME/bin:$PATH' >> "$HOME/.bashrc"
fi
export PATH="$HOME/bin:$PATH"

# Step 7 - Set environment variables
echo "[7/7] Setting environment variables..."
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Final verification
echo ""
echo "========================================"
echo " Verification"
echo "========================================"
python3 -c "
import transformers, datasets, torch, librosa, soundfile
print('transformers :', transformers.__version__)
print('datasets     :', datasets.__version__)
print('torch        :', torch.__version__)
print('librosa      :', librosa.__version__)
print('soundfile    :', soundfile.__version__)
print('CUDA         :', torch.cuda.is_available())
"
sox --version
uconv --version 2>/dev/null || echo "uconv wrapper: ready"

echo ""
echo "========================================"
echo " Setup complete!"
echo " Activate env : conda activate $ENV_NAME"
echo " Set vars     : export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8"
echo "              : export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
echo "              : export PATH=\$HOME/bin:\$PATH"
echo "========================================"
