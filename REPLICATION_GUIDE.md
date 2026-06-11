# Replication Guide

Complete step-by-step instructions to replicate all training and evaluation in this research.

**System Requirements:**
- Ubuntu Linux (tested on Ubuntu 24)
- NVIDIA GPUs with CUDA support (tested on 4x RTX 2080 Ti, 11GB VRAM each)
- Miniconda or Anaconda installed
- No sudo/admin access required
- ~50GB free disk space

**Dataset:**
- UWB-ATCC corpus (Air Traffic Control Communications, Prague Airport)
- Download from: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
- Place the downloaded zip at `~/Downloads/Air Traffic Control Communication.zip`

---

## Part 1: Wav2Vec2 Large (End-to-End CTC Model)

### 1.1 Clone Repository
```bash
cd ~
git clone https://github.com/suideepmax/Pilot-to-ATC-Research.git
```

### 1.2 Clone the IDIAP Base Repository
```bash
cd ~
git clone https://github.com/idiap/w2v2-air-traffic
cd w2v2-air-traffic
```

### 1.3 Create Conda Environment
```bash
conda create -n w2v2_asr -c conda-forge python=3.10 -y
conda activate w2v2_asr
```
The `-c conda-forge` flag is critical — the default conda channel installs GraalPy instead of CPython on some systems.

### 1.4 Setup Environment
```bash
cd ~/w2v2-air-traffic
bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/setup_environment.sh
```
This installs all dependencies from the IDIAP repository's requirements.txt, fixes known issues (pyctcdecode version typo), and creates a uconv wrapper for text normalization (no sudo needed).

### 1.5 Prepare UWB-ATCC Data
```bash
cd ~/w2v2-air-traffic
bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/data_prepare_uwb_atcc.sh
```
This script extracts the UWB-ATCC zip/rar archives, parses TRS transcripts, applies text normalization, and creates an 80/20 train/test split (seed=1234).

Expected output:
- Train: 11,543 utterances, 2,086 recordings
- Test: 2,886 utterances, 570 recordings

### 1.6 Train Wav2Vec2 Large
```bash
cd ~/w2v2-air-traffic
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/train_wav2vec2_large.sh
```

Training details:
- Model: facebook/wav2vec2-large-960h-lv60-self (317M params, 100% trained)
- Steps: 10,000 | LR: 5e-4 | Warmup: 1,000
- Effective batch size: 64 (1/GPU x 4 GPUs x 16 grad accum) — paper used 24 on 1 GPU
- Precision: fp16 mixed | Multi-GPU: DDP
- Estimated time: ~8.6 hours on 4x RTX 2080 Ti

### 1.7 Evaluate Wav2Vec2
```bash
cd ~/w2v2-air-traffic
bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/eval_large_model.sh
```
Expected: WER 14.54% (no LM) / 12.69% (with KenLM 4-gram)

---

## Part 2: Canary-Qwen-2.5B (Hybrid SALM Model)

### 2.1 Create Conda Environment
```bash
conda create -n canary_ft -c conda-forge python=3.11 -y
conda activate canary_ft
```

### 2.2 Install PyTorch 2.6
```bash
nvidia-smi | head -3  # Check CUDA driver version
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2.3 Install NeMo from Trunk
```bash
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```
The pip-released nemo_toolkit does NOT include the speechlm2 module.

### 2.4 Install Dependencies
```bash
pip install lhotse sentencepiece transformers==4.51.0 datasets librosa soundfile
pip install hydra-core omegaconf pytorch-lightning webdataset braceexpand
pip install editdistance jiwer peft==0.14.0
conda install -c conda-forge sox -y
```
Versions pinned: transformers==4.51.0 and peft==0.14.0 for Qwen3/LoRA compatibility.

### 2.5 Verify Installation
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())
import nemo; print('NeMo:', nemo.__version__)
from nemo.collections.speechlm2.models import SALM
print('SALM import: OK')
"
```

### 2.6 Clone NeMo Repository
```bash
git clone --depth 1 https://github.com/NVIDIA/NeMo.git ~/NeMo
```

### 2.7 Convert Data to NeMo Format
```bash
mkdir -p ~/canary-ft/data
python ~/Pilot-to-ATC-Research/models/canary-qwen/scripts/convert_uwb_atcc_to_nemo.py \
    --data-dir ~/w2v2-air-traffic/experiments/data/uwb_atcc \
    --output-dir ~/canary-ft/data \
    --repo-dir ~/w2v2-air-traffic \
    --target-sr 16000
```
Resamples audio from 8kHz to 16kHz and creates NeMo JSONL manifests.

### 2.8 Convert to Lhotse Cuts
```bash
cd ~/canary-ft
python -c "
import lhotse, json
from lhotse import CutSet, MonoCut, SupervisionSegment
for split in ['train', 'test']:
    entries = [json.loads(l) for l in open(f'data/{split}_manifest.json')]
    cuts = [MonoCut(id=e['audio_filepath'].split('/')[-1].replace('.wav',''),
        start=0, duration=e['duration'], channel=0,
        recording=lhotse.Recording.from_file(e['audio_filepath'],
            recording_id=e['audio_filepath'].split('/')[-1].replace('.wav','')),
        supervisions=[SupervisionSegment(
            id=e['audio_filepath'].split('/')[-1].replace('.wav',''),
            recording_id=e['audio_filepath'].split('/')[-1].replace('.wav',''),
            start=0, duration=e['duration'], text=e['text'])])
        for e in entries]
    CutSet.from_cuts(cuts).to_file(f'data/{split}_cuts.jsonl.gz')
    print(f'{split}: {len(cuts)} cuts')
"
```

### 2.9 Setup Training Config
```bash
mkdir -p ~/canary-ft/conf
cp ~/Pilot-to-ATC-Research/models/canary-qwen/scripts/salm_uwb_atcc.yaml ~/canary-ft/conf/
```
**Important**: Edit the cuts_path values in the YAML to match your actual paths.

### 2.10 Train Canary-Qwen (LoRA — 0.97% params)
```bash
cd ~/canary-ft
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc
```
- Trainable: 27.8M / 2,870M (0.97%) via LoRA + modality adapter
- Estimated time: ~5.3 hours on 4x RTX 2080 Ti

### 2.11 Train Canary-Qwen (Encoder Unfrozen — 32.8% params) — Optional
```bash
cp ~/Pilot-to-ATC-Research/models/canary-qwen/scripts/salm_uwb_atcc_unfrozen.yaml ~/canary-ft/conf/
rm -rf ~/canary-ft/experiments/checkpoints/*
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ~/NeMo/examples/speechlm2/salm_train.py \
    --config-path=/home/kotasthane/canary-ft/conf \
    --config-name=salm_uwb_atcc_unfrozen
```
- Trainable: 838.8M / 2,870M (32.8%)
- Estimated time: ~5.3 hours

### 2.12 Evaluate Fine-Tuned Model
```bash
cd ~/canary-ft
CUDA_VISIBLE_DEVICES=0 python -c "
import torch, json
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from nemo.collections.speechlm2.models import SALM
from jiwer import wer
dcp_to_torch_save('experiments/checkpoints/step=10000-last.ckpt', '/tmp/consolidated.pt')
model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
state = torch.load('/tmp/consolidated.pt', map_location='cpu', weights_only=False)
if 'state_dict' in state: state = state['state_dict']
model.load_state_dict(state, strict=False)
model.cuda().eval()
samples = [json.loads(l) for l in open('data/test_manifest.json')]
refs, hyps = [], []
for s in samples:
    ids = model.generate(prompts=[[{'role':'user',
        'content':f'Transcribe the following: {model.audio_locator_tag}',
        'audio':[s['audio_filepath']]}]], max_new_tokens=128)
    refs.append(s['text'].lower().strip())
    hyps.append(model.tokenizer.ids_to_text(ids[0].cpu()).lower().strip())
print(f'WER: {wer(refs, hyps)*100:.2f}%')
"
```
Expected: LoRA = 23.32% WER | Encoder unfrozen = 23.82% WER | v3 (LoRA + SpecAugment + regularization) = 20.70% WER

---

## Part 3: Known Issues and Fixes

| Issue | Fix |
|---|---|
| Conda installs GraalPy | Use `-c conda-forge` |
| fsspec 2026.x breaks HuggingFace | Pin `fsspec==2023.6.0` in W2V2 env |
| fp16 + AdamW eps=1e-8 causes NaN | Set `eps: 1e-4` in optimizer config |
| "Too many open files" crash | `ulimit -n 65536` + `num_workers: 1` |
| DDP OOM with 2.5B model | Use ModelParallelStrategy (FSDP) |
| ModelParallelStrategy rejects 16-mixed | Use `precision: 16-true` |
| NeMo FSDP doesn't log metrics | Extract val_loss from checkpoint messages |
| CUDA init fails with faulty GPU | Wait for other jobs or exclude GPU via CUDA_VISIBLE_DEVICES |
| Unfreezing encoder doesn't improve WER | Frozen LLM decoder is the bottleneck |

---

## Part 4: Results Summary (UWB-ATCC)

| Model | Params Trained | WER | Time |
|---|---|---|---|
| W2V2 Large (no LM) | 317M (100%) | 14.54% | ~8.6 hrs |
| W2V2 Large (with KenLM) | 317M (100%) | 12.69% | ~8.6 hrs |
| Canary-Qwen v3 (LoRA + SpecAugment) | 27.8M (0.97%) | 20.70% | ~5.3 hrs |
| Canary-Qwen LoRA (no regularization) | 27.8M (0.97%) | 23.32% | ~5.3 hrs |
| Canary-Qwen Unfrozen | 838.8M (32.8%) | 23.82% | ~5.3 hrs |
| Canary-Qwen Zero-Shot | 0 | 81.49% | N/A |

All models: 10,000 steps, lr=5e-4, warmup=1,000, 4x RTX 2080 Ti.

HuggingFace models:
- W2V2: https://huggingface.co/suideepmax/wav2vec2-large-960h-lv60-self-atc-uwb-atcc
- Canary LoRA: https://huggingface.co/suideepmax/canary-qwen-2.5b-atc-lora
- Canary Unfrozen: https://huggingface.co/suideepmax/canary-qwen-2.5b-atc-unfrozen

---

## Part 5: ATCOSIM Corpus — Wav2Vec2 Large

**Prerequisites:** Complete Part 1 (W2V2 environment setup) first.

### 5.1 Prepare ATCOSIM Data

```bash
cd ~/w2v2-air-traffic
bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/data_prepare_atcosim.sh
```

This downloads the ATCOSIM ISO (~2.5GB), extracts it with `bsdtar`, and runs the full
normalization + train/test split pipeline. Output:
- `experiments/data/atcosim_corpus/{train,test}/`
- Gender subsets: `{train,test}_{female,male}/`

### 5.2 Apply the Same Script Modifications as UWB-ATCC

The same DDP/batch-size fixes from Part 1.4 apply. If you already ran the UWB-ATCC
training, these changes are already in place in `src/run_asr_fine_tuning.sh` and
`ablations/atcosim/train_w2v2_large-60v.sh`.

```bash
# Reduce batch size to fit 317M model on 11GB VRAM
sed -i 's/per_device_train_batch_size=16/per_device_train_batch_size=1/' ablations/atcosim/train_w2v2_large-60v.sh
sed -i 's/gradient_acc=2/gradient_acc=16/' ablations/atcosim/train_w2v2_large-60v.sh
```

### 5.3 Train

```bash
cd ~/w2v2-air-traffic
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ~/Pilot-to-ATC-Research/models/w2v2/scripts/train_wav2vec2_atcosim_large.sh
```

- Steps: 5,000 | LR: 5e-4 | Warmup: 1,000
- Effective batch size: 64 (1/GPU x 4 GPUs x 16 grad accum)
- Expected time: TBD

### 5.4 Results
See `models/w2v2/docs/PROGRESS_ATCOSIM.md` for results as they come in.
