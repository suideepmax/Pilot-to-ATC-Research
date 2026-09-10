#!/usr/bin/env python3
"""Evaluate fine-tuned Canary-Qwen checkpoint (FSDP distributed checkpoint).
Usage: CUDA_VISIBLE_DEVICES=0 python eval_finetuned.py \
    --checkpoint ~/canary-ft/experiments/checkpoints/step=10000-last.ckpt \
    --test-manifest ~/canary-ft/data/test_manifest.json
"""
import argparse, hashlib, json, os, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from jiwer import wer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='eval_results.json')
    p.add_argument('--max-samples', type=int, default=0)
    args = p.parse_args()

    # Cache path is derived from the checkpoint path (not a fixed global path) so
    # that concurrent evaluations of different checkpoints never collide and
    # silently reuse the wrong weights. (Bug found and fixed 2026-09-08: a
    # fixed '/tmp/canary_eval_consolidated.pt' path caused two evals run in
    # parallel to load identical weights — see research_log/VALIDATION.md VAL-012.)
    ckpt_hash = hashlib.sha256(os.path.abspath(args.checkpoint).encode()).hexdigest()[:16]
    consolidated = f'/tmp/canary_eval_consolidated_{ckpt_hash}.pt'
    if not os.path.exists(consolidated):
        print(f'Consolidating {args.checkpoint} -> {consolidated}...')
        dcp_to_torch_save(args.checkpoint, consolidated)
    else:
        print(f'Reusing cached consolidation for this exact checkpoint: {consolidated}')

    state = torch.load(consolidated, map_location='cpu', weights_only=False)
    if 'state_dict' in state: state = state['state_dict']
    nans = sum(1 for v in state.values() if torch.isnan(v).any())
    print(f'NaN check: {nans}/{len(state)}')
    assert nans == 0, 'NaN weights detected — check AdamW eps setting'

    from nemo.collections.speechlm2.models import SALM
    print('Loading model...')
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    # ISS-013 (2026-09-10): strict=False was previously silently discarding
    # missing/unexpected keys. The base model here is the RELEASED
    # canary-qwen-2.5b (LoRA-shaped LLM keys, e.g.
    # llm.base_model.model.model.layers.*), but a non-LoRA full-decoder-FT
    # checkpoint (e.g. S3-B3) has plain llm.model.layers.* keys -- a
    # structural mismatch under which strict=False would let the ENTIRE LLM
    # silently fail to load, and this script would then report the
    # RELEASED model's WER as if it were the trained checkpoint's. Assert on
    # the actual incompatibility lists instead of discarding them.
    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing keys, {n_unexpected} unexpected keys')
    if n_missing:
        print(f'  missing (first 10): {load_result.missing_keys[:10]}')
    if n_unexpected:
        print(f'  unexpected (first 10): {load_result.unexpected_keys[:10]}')
    assert n_missing + n_unexpected <= 5, (
        f'ISS-013: {n_missing} missing + {n_unexpected} unexpected keys when loading '
        f'{args.checkpoint} onto nvidia/canary-qwen-2.5b -- this looks like a structural '
        f'mismatch (e.g. LoRA-shaped base model vs a plain full-parameter checkpoint), not '
        f'a handful of harmless buffer differences. Evaluating anyway would silently report '
        f"the BASE model's performance, not this checkpoint's. See research_log/ISSUES.md ISS-013."
    )
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
