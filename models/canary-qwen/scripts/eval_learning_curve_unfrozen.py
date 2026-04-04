#!/usr/bin/env python3
"""Learning curve for encoder-unfrozen Canary-Qwen run."""

import torch, json, os
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from nemo.collections.speechlm2.models import SALM
from jiwer import wer

CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 7500, 10000]
SAMPLE_SIZE = 500

with open('data/test_manifest.json') as f:
    samples = [json.loads(line) for line in f][:SAMPLE_SIZE]

print(f"Evaluating {len(samples)} samples at each checkpoint\n")

print("Loading base model...")
model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')

results = []

for step in CHECKPOINTS:
    ckpt_dir = f'experiments/checkpoints/step={step}.ckpt'
    tmp_path = f'/tmp/canary_unfrozen_step{step}.pt'

    if not os.path.isdir(ckpt_dir):
        print(f"Step {step}: not found, skipping")
        continue

    if not os.path.exists(tmp_path):
        print(f"Step {step}: consolidating...")
        dcp_to_torch_save(ckpt_dir, tmp_path)

    state = torch.load(tmp_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state: state = state['state_dict']

    nans = sum(1 for v in state.values() if torch.isnan(v).any())
    if nans > 0:
        print(f"Step {step}: {nans} NaN keys, SKIPPING")
        results.append({"step": step, "wer": None, "nan": True})
        continue

    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    refs, hyps = [], []
    for s in samples:
        try:
            answer_ids = model.generate(
                prompts=[[{
                    'role': 'user',
                    'content': f'Transcribe the following: {model.audio_locator_tag}',
                    'audio': [s['audio_filepath']]
                }]],
                max_new_tokens=128,
            )
            pred = model.tokenizer.ids_to_text(answer_ids[0].cpu())
            refs.append(s['text'].lower().strip())
            hyps.append(pred.lower().strip())
        except:
            pass

    step_wer = wer(refs, hyps)
    results.append({"step": step, "wer": step_wer, "samples": len(refs)})
    print(f"Step {step:>5d}: WER = {step_wer*100:.2f}%")

print(f"\n{'='*60}")
print("LEARNING CURVE: ENCODER UNFROZEN (32.8% params)")
print(f"{'='*60}")
print(f"{'Step':>6s}  {'Unfrozen':>10s}  {'LoRA-only':>10s}  {'W2V2':>10s}")
print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

lora_wers = {500: 39.14, 1000: 45.02, 2000: 30.87, 3000: 26.28, 5000: 24.77, 7500: 24.53, 10000: 24.53}
w2v2_wers = {500: 26.80, 1000: 22.70, 2000: 19.13, 3000: 17.86, 5000: 17.42, 7500: 15.92, 10000: 15.15}

for r in results:
    step = r['step']
    if r.get('nan'):
        uf = 'NaN'
    else:
        uf = f"{r['wer']*100:.2f}%"
    lo = f"{lora_wers.get(step, 0):.2f}%" if step in lora_wers else "-"
    w2 = f"{w2v2_wers.get(step, 0):.2f}%" if step in w2v2_wers else "-"
    print(f"{step:>6d}  {uf:>10s}  {lo:>10s}  {w2:>10s}")

with open('learning_curve_unfrozen.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to learning_curve_unfrozen.json")
