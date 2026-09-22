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
    # ISS-013 fix, ported from eval_finetuned.py (2026-09-22): this script was
    # only ever run on v1/v3 (LoRA adapters trained on top of the released
    # nvidia/canary-qwen-2.5b checkpoint), so hardcoding SALM.from_pretrained
    # as the base was correct for those. A full-decoder-FT or independently
    # LoRA-trained checkpoint (e.g. the matched-protocol full-decoder/LoRA
    # arms) needs the exact architecture salm_train_stable.py trained
    # (--base composed) or load_state_dict(strict=False) silently discards
    # most of the trained weights instead of erroring.
    p.add_argument('--base', choices=['released', 'composed'], default='released')
    p.add_argument('--exp-config', default=None,
                    help="Required when --base composed.")
    args = p.parse_args()
    if args.base == 'composed' and not args.exp_config:
        p.error('--exp-config is required when --base composed')

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
    print(f'Loading model (base={args.base})...')
    if args.base == 'released':
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        base_desc = 'nvidia/canary-qwen-2.5b (released)'
    else:
        from omegaconf import OmegaConf
        exp_cfg = OmegaConf.load(args.exp_config)
        model = SALM(OmegaConf.to_container(exp_cfg.model, resolve=True))
        # Composed construction defaults to fp32; checkpoint tensors are fp16
        # (precision: 16-true) -- cast to match, same as eval_finetuned.py.
        model = model.half()
        base_desc = (f"composed from {args.exp_config} "
                     f"(pretrained_llm={exp_cfg.model.pretrained_llm}, "
                     f"pretrained_asr={exp_cfg.model.pretrained_asr})")

    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing keys, {n_unexpected} unexpected keys')
    if n_missing:
        print(f'  missing (first 10): {load_result.missing_keys[:10]}')
    if n_unexpected:
        print(f'  unexpected (first 10): {load_result.unexpected_keys[:10]}')
    max_allowed_mismatch = 0 if args.base == 'composed' else 5
    assert n_missing + n_unexpected <= max_allowed_mismatch, (
        f'ISS-013: {n_missing} missing + {n_unexpected} unexpected keys when loading '
        f'{args.checkpoint} onto {base_desc} -- structural mismatch, would silently '
        f"report the BASE model's performance, not this checkpoint's."
    )
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Generating {args.num_beams}-best for {len(samples)} samples...')

    # Determinism fix, ported from eval_finetuned.py's ISS-013 follow-up
    # (2026-09-22): passing do_sample=False (or any field matching
    # GenerationConfig()'s own class default) NESTED inside a GenerationConfig
    # object does NOT reliably stick for --base composed checkpoints --
    # transformers backfills any field left equal to the class default from
    # the underlying LLM's own generation_config.json (Qwen3-1.7B's is
    # do_sample=True/temperature=0.6/top_k=20/top_p=0.95), silently turning
    # intended beam search into sampling. Passing these as direct **kwargs to
    # .generate() applies after that merge and reliably wins -- verified for
    # eval_finetuned.py via a repeat-run determinism check (VAL/ISS-013); the
    # same fix is required here since this script is now also used with
    # --base composed, which it never was before.
    gen_kwargs = dict(
        bos_token_id=model.text_bos_id,
        eos_token_id=model.text_eos_id,
        pad_token_id=model.text_pad_id,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=False,
        temperature=None, top_k=None, top_p=None,
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
                    max_new_tokens=128, **gen_kwargs)
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
