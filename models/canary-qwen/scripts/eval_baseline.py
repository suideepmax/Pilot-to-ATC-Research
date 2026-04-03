
#!/usr/bin/env python3

"""Evaluate pretrained Canary-Qwen-2.5B on UWB-ATCC test set (zero-shot baseline)."""

import json, torch

from jiwer import wer

print("Loading model...")

from nemo.collections.speechlm2.models import SALM

model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')

model = model.cuda()

model.eval()

print("Model loaded")

with open('data/test_manifest.json') as f:

    samples = [json.loads(line) for line in f]

print(f"Evaluating {len(samples)} samples...")

refs, hyps = [], []

errors = 0

for i, s in enumerate(samples):

    if (i+1) % 100 == 0:

        running_wer = wer(refs, hyps) if refs else 0

        print(f"  {i+1}/{len(samples)}  running WER: {running_wer*100:.1f}%")

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

    except Exception as e:

        errors += 1

final_wer = wer(refs, hyps)

print(f"\n{'='*50}")

print(f"BASELINE RESULTS (zero-shot, no fine-tuning)")

print(f"  Samples evaluated: {len(refs)}")

print(f"  Errors: {errors}")

print(f"  WER: {final_wer*100:.2f}%")

print(f"{'='*50}")

with open('baseline_results.json', 'w') as f:

    json.dump({'wer': final_wer, 'num_samples': len(refs), 'errors': errors}, f, indent=2)

print("Results saved to baseline_results.json")

