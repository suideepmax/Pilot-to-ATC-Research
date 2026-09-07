# Environment

## ENV-001 — System hardware

Date recovered: 2026-09-06
Source: README.md, SUMMARY.md, REPLICATION_GUIDE.md, models/canary-qwen/docs/SETUP.md

- Machine: "ET335Lambda" Ubuntu workstation (README.md)
- GPUs: 4x NVIDIA RTX 2080 Ti, 11GB VRAM each
- No sudo/admin access on the machine
- CUDA driver verified for Canary-Qwen setup: 570.207, CUDA 12.8 (models/canary-qwen/docs/SETUP.md)
- OS tested: Ubuntu 24 (models/w2v2/docs/SETUP.md, REPLICATION_GUIDE.md)

Related Records: [[ENV-002]], [[ENV-003]]

---

## ENV-002 — W2V2 (idiap/w2v2-air-traffic) conda environment `w2v2_asr`

Date recovered: 2026-09-06
Source: models/w2v2/docs/SETUP.md, REPLICATION_GUIDE.md

Verified package versions (as recorded in SETUP.md "Verified Package Versions" table):

| Package | Version |
|---|---|
| Python | 3.10 (CPython, via conda-forge) |
| torch | 1.13.0+cu117 |
| transformers | 4.24.0 |
| datasets | 2.14.0 |
| pyarrow | 14.0.1 |
| fsspec | 2023.6.0 |
| librosa | 0.9.2 |
| soundfile | 0.13.1 |
| setuptools | 67.6.0 |
| sox | 14.4.2 |

Known constraints/fixes recorded in SETUP.md:
1. Default conda channel installs GraalPy (JVM-based Python) instead of CPython, breaking numpy — must create env with `-c conda-forge`.
2. `requirements.txt` has a typo: `pyctcdecode=0.4.0` (single `=`, invalid for pip) — fixed to `==0.4.0`.
3. fsspec 2026.x breaks local dataset loading (`is_remote_filesystem()` misbehaves) — pin `fsspec==2023.6.0`.
4. Latest pyarrow breaks `datasets` arrow_dataset imports — pin `pyarrow==14.0.1`, `datasets==2.14.0`.
5. `uconv` (ICU tool, needed for .trs CP1250→UTF-8 conversion) requires sudo to install via icu-devtools — replaced with a Python drop-in wrapper (`scripts/uconv_wrapper.py`) installed at `~/bin/uconv`.
6. librosa 0.8.1 imports `pkg_resources`, which needs a specific setuptools — pin `setuptools==67.6.0`, `librosa==0.9.2`.
7. sox required for wav.scp audio loading but not in requirements.txt — install via `conda install -c conda-forge sox`.
8. `datasets` cache_dir triggers a LocalFileSystem bug in `builder.py` — patched to use `/tmp/hf_cache_{train,eval}` in `src/run_speech_recognition_ctc.py`.

Required session environment variables (SETUP.md):
```
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)   # run from w2v2-air-traffic repo root
export PATH=$HOME/bin:$PATH
```

Related Records: [[ENV-001]], [[AUD-002]]

---

## ENV-003 — Canary-Qwen (NeMo speechlm2) conda environment `canary_ft`

Date recovered: 2026-09-06
Source: models/canary-qwen/docs/SETUP.md, models/canary-qwen/docs/PROGRESS.md, REPLICATION_GUIDE.md

- Conda env: `canary_ft`, Python 3.11
- PyTorch 2.6.0+cu124 (FSDP2 support required) — installed via `--index-url https://download.pytorch.org/whl/cu124`
- NeMo installed from GitHub trunk (`pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"`) — the pip-released `nemo_toolkit` does NOT include the `speechlm2` module needed for SALM
- NeMo version recorded during training: 2.8.0rc0 (models/w2v2 PROGRESS_ATCOSIM.md / canary PROGRESS.md)
- Additional deps: lhotse, sentencepiece, transformers==4.51.0, datasets, librosa, soundfile, hydra-core, omegaconf, pytorch-lightning, webdataset, braceexpand, editdistance, jiwer, peft==0.14.0, sox (conda-forge)
- transformers==4.51.0 and peft==0.14.0 pinned specifically for Qwen3/LoRA compatibility (REPLICATION_GUIDE.md 2.4)
- Distributed strategy used: FSDP via `ModelParallelStrategy` (tensor_parallel=1, data_parallel=4) — plain DDP OOMs on the 2.5B model across 11GB GPUs
- Precision: fp16-true, with AdamW `eps=1e-4` (required for numerical stability; default 1e-8 and even research-suggested 1e-6 caused NaN — see ISS-003)

Known issues (models/canary-qwen/docs/SETUP.md, PROGRESS.md, REPLICATION_GUIDE.md):
- GPU left in an error state by a concurrent/crashed job poisons the CUDA context system-wide; must wait for other jobs to clear or exclude the bad GPU via `CUDA_VISIBLE_DEVICES`.
- NeMo trunk install may have dependency conflicts — install in a fresh conda env.
- Canary requires 16kHz mono audio; UWB-ATCC source is 8kHz — resampling required during data conversion (`convert_uwb_atcc_to_nemo.py --target-sr 16000`).
- `ModelParallelStrategy` rejects `16-mixed` precision — must use `16-true`.
- "Too many open files" crash — fixed with `ulimit -n 65536` + `num_workers: 1`.
- NeMo's FSDP path does not log metrics to stdout in the usual way — val_loss had to be extracted from checkpoint messages.

Related Records: [[ENV-001]], [[ISS-003]], [[AUD-003]]

---

## ENV-004 — No tmux/screen on this Lambda host; no sudo to install; conda env must be explicitly activated for detached/non-interactive launches

Date: 2026-09-07
Status: OPEN (informational — durable host constraint)

Description: `tmux` and `screen` are both absent (`command not found`) on this machine, and the user has no sudo (per [[ENV-001]]), so neither can be installed via apt. `nohup`/`setsid` are available and were used as the persistent-session fallback for EXP-007's training-monitor workflow. Separately: a detached `setsid nohup bash -c '...'` shell does NOT inherit the interactive shell's conda environment or exported variables (e.g. `PYTHONPATH`) — it must explicitly `source ~/miniconda3/etc/profile.d/conda.sh && conda activate w2v2_asr` and set any variables the target script expects, or scripts written assuming an already-activated interactive shell (like `ablations/atcosim/train_w2v2_large-60v*.sh` and `src/run_asr_fine_tuning.sh`, which do `export PYTHONPATH=$PYTHONPATH:$(pwd)` under `set -u`) will fail immediately or run under the wrong (`base`) conda env silently.

Evidence: `which tmux screen` → not found; `which nohup setsid` → present; first EXP-007 female launch attempt failed twice for exactly this reason (`PYTHONPATH: unbound variable`, then `ModuleNotFoundError: No module named 'datasets'` under base env's python3.13) before being corrected — see [[ISS-006]].

Impact: Any future long-running job launched via a detached/non-interactive shell on this host must explicitly activate the correct conda env and set `PYTHONPATH` (or any other variable the target script assumes) — this is not optional and is easy to get silently wrong given [[ISS-006]]'s discovery that the wrapper scripts don't fail loudly when it happens.

Mitigation: Standard detached-launch template for this host:
```
setsid nohup bash -c '
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env_name>
export PYTHONPATH=
cd <repo_dir>
bash <target_script>
' > <logfile> 2>&1 < /dev/null &
disown
```
Monitoring uses the `Monitor`/`tail -F` mechanism against the redirected log file in place of a tmux monitoring pane.

Related Records: [[ENV-001]], [[ENV-002]], [[ISS-006]]
