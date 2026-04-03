#!/usr/bin/env python3
"""Evaluate Canary-Qwen at multiple checkpoints to build WER learning curve."""

import torch, json, os
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from nemo.collections.speechlm2.models import SALM
from jiwer import wer

CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 7500, 10000]
SAMPLE_SIZE = 500  # evaluate on 500 samples per checkpoint for speed

# Load test data
with open('data/test_manifest.json') as f:
    all_samples = [json.loads(line) for line in f]
samples = all_samples[:SAMPLE_SIZE]
print(f"Evaluating {len(samples)} samples at each checkpoint\n")

# Load base model once
print("Loading base model...")
base_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
base_state = base_model.state_dict()

results = []

for step in CHECKPOINTS:
    ckpt_dir = f'experiments/checkpoints/step={step}.ckpt'
    tmp_path = f'/tmp/canary_step{step}.pt'

    if not os.path.isdir(ckpt_dir):
        print(f"Step {step}: checkpoint not found, skipping")
        continue

    # Consolidate
    if not os.path.exists(tmp_path):
        print(f"Step {step}: consolidating...")
        dcp_to_torch_save(ckpt_dir, tmp_path)

    # Load weights
    state = torch.load(tmp_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state: state = state['state_dict']

    nans = sum(1 for v in state.values() if torch.isnan(v).any())
    if nans > 0:
        print(f"Step {step}: {nans} NaN keys, SKIPPING")
        results.append({"step": step, "wer": None, "nan": True})
        continue

    base_model.load_state_dict(state, strict=False)
    base_model.cuda().eval()

    # Evaluate
    refs, hyps = [], []
    for i, s in enumerate(samples):
        try:
            answer_ids = base_model.generate(
                prompts=[[{
                    'role': 'user',
                    'content': f'Transcribe the following: {base_model.audio_locator_tag}',
                    'audio': [s['audio_filepath']]
                }]],
                max_new_tokens=128,
            )
            pred = base_model.tokenizer.ids_to_text(answer_ids[0].cpu())
            refs.append(s['text'].lower().strip())
            hyps.append(pred.lower().strip())
        except:
            pass

    step_wer = wer(refs, hyps)
    results.append({"step": step, "wer": step_wer, "samples": len(refs)})
    print(f"Step {step:>5d}: WER = {step_wer*100:.2f}%  ({len(refs)} samples)")

print(f"\n{'='*50}")
print("LEARNING CURVE SUMMARY")
print(f"{'='*50}")
print(f"{'Step':>6s}  {'WER':>8s}")
print(f"{'-'*6}  {'-'*8}")
for r in results:
    if r.get('nan'):
        print(f"{r['step']:>6d}  {'NaN':>8s}")
    else:
        print(f"{r['step']:>6d}  {r['wer']*100:>7.2f}%")
print(f"\nBaseline (zero-shot): 81.49%")
print(f"W2V2 large (no LM):  14.54%")
print(f"W2V2 large (with LM): 12.69%")

with open('learning_curve.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to learning_curve.json")
