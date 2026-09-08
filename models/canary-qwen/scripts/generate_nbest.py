#!/usr/bin/env python3
"""Generate N-best beam-search hypotheses from a fine-tuned Canary-Qwen checkpoint.
Part of the decoding-fairness comparison (S4-FAIR): produces N-best + per-sequence
model scores so a separate script (rescore_kenlm.py, run in an env with kenlm
bindings) can rescore with the same in-domain KenLM used for the W2V2 baseline.

Usage: CUDA_VISIBLE_DEVICES=0 python generate_nbest.py \
    --checkpoint ~/canary-ft/experiments/checkpoints/step=10000-last.ckpt \
    --test-manifest ~/canary-ft/data/test_manifest.json \
    --output nbest_v1.json --num-beams 5
"""
import argparse, hashlib, json, os, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='nbest_results.json')
    p.add_argument('--num-beams', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=0)
    args = p.parse_args()

    # Same hash-derived cache path fix as eval_finetuned.py (VAL-012) — avoids
    # collisions between concurrent evaluations of different checkpoints.
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
    assert nans == 0, 'NaN weights detected'

    from nemo.collections.speechlm2.models import SALM
    from transformers import GenerationConfig
    print('Loading model...')
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Generating {args.num_beams}-best for {len(samples)} samples...')

    gen_cfg = GenerationConfig(
        bos_token_id=model.text_bos_id,
        eos_token_id=model.text_eos_id,
        pad_token_id=model.text_pad_id,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=False,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    results, errors = [], 0
    with torch.no_grad():
        for i, s in enumerate(samples):
            if (i + 1) % 200 == 0:
                print(f'  {i+1}/{len(samples)} ({errors} errors so far)')
            try:
                out = model.generate(
                    prompts=[[{'role': 'user',
                        'content': f'Transcribe the following: {model.audio_locator_tag}',
                        'audio': [s['audio_filepath']]}]],
                    generation_config=gen_cfg, max_new_tokens=128)
                seqs = out.sequences.cpu()
                scores = out.sequences_scores.cpu().tolist()
                hyps = [model.tokenizer.ids_to_text(seq).lower().strip() for seq in seqs]
                results.append({
                    'audio_filepath': s['audio_filepath'],
                    'text': s['text'].lower().strip(),
                    'hyps': [{'text': h, 'model_score': sc} for h, sc in zip(hyps, scores)],
                })
            except Exception as e:
                errors += 1
                print(f'  [error @ {i}] {e}')

    print(f'\nDone: {len(results)} samples, {errors} errors')
    json.dump({'num_beams': args.num_beams, 'errors': errors, 'results': results},
               open(args.output, 'w'), indent=2)
    print(f'Saved to {args.output}')

if __name__ == '__main__': main()
