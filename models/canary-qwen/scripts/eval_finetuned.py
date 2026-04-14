#!/usr/bin/env python3
"""Evaluate fine-tuned Canary-Qwen checkpoint (FSDP distributed checkpoint).
Usage: CUDA_VISIBLE_DEVICES=0 python eval_finetuned.py \
    --checkpoint ~/canary-ft/experiments/checkpoints/step=10000-last.ckpt \
    --test-manifest ~/canary-ft/data/test_manifest.json
"""
import argparse, json, os, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from jiwer import wer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='eval_results.json')
    p.add_argument('--max-samples', type=int, default=0)
    args = p.parse_args()

    consolidated = '/tmp/canary_eval_consolidated.pt'
    if not os.path.exists(consolidated):
        print(f'Consolidating {args.checkpoint}...')
        dcp_to_torch_save(args.checkpoint, consolidated)

    state = torch.load(consolidated, map_location='cpu', weights_only=False)
    if 'state_dict' in state: state = state['state_dict']
    nans = sum(1 for v in state.values() if torch.isnan(v).any())
    print(f'NaN check: {nans}/{len(state)}')
    assert nans == 0, 'NaN weights detected — check AdamW eps setting'

    from nemo.collections.speechlm2.models import SALM
    print('Loading model...')
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Evaluating {len(samples)} samples...')

    refs, hyps, errors = [], [], 0
    for i, s in enumerate(samples):
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(samples)} WER: {wer(refs, hyps)*100:.1f}%')
        try:
            ids = model.generate(prompts=[[{'role':'user',
                'content':f'Transcribe the following: {model.audio_locator_tag}',
                'audio':[s['audio_filepath']]}]], max_new_tokens=128)
            refs.append(s['text'].lower().strip())
            hyps.append(model.tokenizer.ids_to_text(ids[0].cpu()).lower().strip())
        except: errors += 1

    final_wer = wer(refs, hyps)
    print(f'\nWER: {final_wer*100:.2f}% ({len(refs)} samples, {errors} errors)')
    json.dump({'wer': final_wer, 'samples': len(refs), 'errors': errors},
              open(args.output, 'w'), indent=2)

if __name__ == '__main__': main()
