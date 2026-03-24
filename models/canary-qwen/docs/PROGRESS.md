# Canary-Qwen-2.5B Fine-Tuning Progress

## Phase 1 - Environment Setup [DONE]
- Conda env `canary_ft` created (Python 3.11)
- PyTorch 2.6.0+cu124 installed
- NeMo 2.8.0rc0 installed from trunk
- SALM import verified
- CUDA verification pending (GPU 1 in error state from concurrent W2V2 training)

## Phase 2 - Data Conversion [IN PROGRESS]
- Input: UWB-ATCC Kaldi format (wav.scp, segments, text)
  - Train: 11,543 utterances
  - Test: 2,886 utterances
  - Sample rate: 8kHz
- Output: NeMo JSONL manifests + individual 16kHz WAV segments
- Script: `scripts/convert_uwb_atcc_to_nemo.py`

## Phase 3 - Baseline Inference [PENDING]
- Run pretrained Canary-Qwen-2.5B on UWB-ATCC test set (zero-shot)
- Measure baseline WER before fine-tuning

## Phase 4 - Fine-Tuning [PENDING]
- Training approach: LoRA on LLM + full encoder + projection training
- Framework: NeMo speechlm2 SALM pipeline
- Config: Based on examples/speechlm2/conf/salm.yaml
- Hardware: 4x RTX 2080 Ti with FSDP

## Phase 5 - Evaluation [PENDING]
- Compare fine-tuned WER vs zero-shot baseline
- Compare with W2V2 results on same dataset
