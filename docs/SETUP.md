# Setup Guide

## Prerequisites
- Miniconda installed (no sudo required)
- CUDA 11.7

## Steps

### 1. Clone the base repo
git clone https://github.com/idiap/w2v2-air-traffic
cd w2v2-air-traffic

### 2. Create conda environment (use conda-forge to avoid GraalPy)
conda create -n w2v2_asr -c conda-forge python=3.10 -y
conda activate w2v2_asr

### 3. Fix requirements typo and install
sed -i 's/pyctcdecode=0\.4\.0/pyctcdecode==0.4.0/' requirements.txt
pip install -r requirements.txt

### 4. Install uconv wrapper (no sudo needed)
mkdir -p ~/bin
# See scripts/install_uconv_wrapper.sh

### 5. Set environment variables (every session)
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

## Known Issues
- Default conda channel pulls GraalPy instead of CPython. Always use -c conda-forge
- requirements.txt has typo: pyctcdecode=0.4.0 should be pyctcdecode==0.4.0
- uconv not available without sudo; use the Python wrapper in scripts/
