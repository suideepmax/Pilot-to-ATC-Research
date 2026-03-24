# Environment Setup Guide

## Prerequisites
- Miniconda installed (no sudo required)
- CUDA 11.7
- Ubuntu 24

## Quick Start (one command)
```bash
bash scripts/setup_environment.sh [env_name] [install_dir]
# Example:
bash scripts/setup_environment.sh w2v2_asr $HOME
```

## What the script does
1. Clones the idiap/w2v2-air-traffic repo
2. Accepts conda Terms of Service
3. Creates a CPython 3.10 conda env via conda-forge (important: default channel gives GraalPy which breaks numpy)
4. Fixes typo in requirements.txt (`pyctcdecode=0.4.0` → `pyctcdecode==0.4.0`)
5. Installs all dependencies
6. Pins compatible package versions (see Known Issues below)
7. Installs uconv Python wrapper
8. Verifies installation

## Known Issues and Fixes

### 1. GraalPy instead of CPython
**Problem:** Default conda channel installs GraalPy (JVM-based Python) which breaks numpy C extensions.
**Fix:** Always use `-c conda-forge` when creating the env.
```bash
conda create -n w2v2_asr -c conda-forge python=3.10 -y
```

### 2. requirements.txt typo
**Problem:** `pyctcdecode=0.4.0` uses single `=` which is invalid for pip.
**Fix:**
```bash
sed -i 's/pyctcdecode=0\.4\.0/pyctcdecode==0.4.0/' requirements.txt
```

### 3. fsspec version conflict
**Problem:** fsspec 2026.x causes `is_remote_filesystem()` to return `True` for local paths, breaking dataset loading.
**Fix:**
```bash
pip install fsspec==2023.6.0
```

### 4. pyarrow incompatibility with datasets
**Problem:** Latest pyarrow breaks `datasets` arrow_dataset imports.
**Fix:**
```bash
pip install pyarrow==14.0.1 datasets==2.14.0
```

### 5. uconv not available without sudo
**Problem:** `uconv` (from icu-devtools) is required for .trs transcript encoding conversion (CP1250 → UTF-8) but needs admin to install.
**Fix:** A Python-based drop-in replacement is provided in `scripts/uconv_wrapper.py` and installed to `~/bin/uconv` by the setup script.

### 6. librosa pkg_resources error
**Problem:** librosa 0.8.1 imports `pkg_resources` which requires a specific setuptools version.
**Fix:**
```bash
pip install setuptools==67.6.0 librosa==0.9.2
```

### 7. sox not available
**Problem:** sox is required for audio loading via wav.scp but not in requirements.txt.
**Fix:**
```bash
conda install -c conda-forge sox -y
```

### 8. datasets LocalFileSystem bug
**Problem:** `datasets` caches to a path that triggers a LocalFileSystem error in `builder.py`.
**Fix:** Patch `src/run_speech_recognition_ctc.py` cache_dir:
```bash
sed -i 's|cache_dir = f".cache/{training_args.output_dir}/train"|cache_dir = "/tmp/hf_cache_train"|' src/run_speech_recognition_ctc.py
sed -i 's|cache_dir = f".cache/{training_args.output_dir}/eval"|cache_dir = "/tmp/hf_cache_eval"|' src/run_speech_recognition_ctc.py
```

## Environment Variables (set every session)
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)  # run from repo root
export PATH=$HOME/bin:$PATH
```

## Verified Package Versions
| Package | Version |
|---|---|
| Python | 3.10 (CPython) |
| torch | 1.13.0+cu117 |
| transformers | 4.24.0 |
| datasets | 2.14.0 |
| pyarrow | 14.0.1 |
| fsspec | 2023.6.0 |
| librosa | 0.9.2 |
| soundfile | 0.13.1 |
| setuptools | 67.6.0 |
| sox | 14.4.2 |
