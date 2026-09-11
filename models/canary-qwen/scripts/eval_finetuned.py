#!/usr/bin/env python3
"""Evaluate fine-tuned Canary-Qwen checkpoint (FSDP distributed checkpoint).
Usage: CUDA_VISIBLE_DEVICES=0 python eval_finetuned.py \
    --checkpoint ~/canary-ft/experiments/checkpoints/step=10000-last.ckpt \
    --test-manifest ~/canary-ft/data/test_manifest.json
"""
import argparse, hashlib, json, os, traceback, torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from jiwer import wer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--output', default='eval_results.json')
    p.add_argument('--max-samples', type=int, default=0)
    # ISS-013 fix (2026-09-10): 'released' (default) preserves the original
    # behavior for v1/v2/v3 -- those are LoRA adapters trained ON TOP OF the
    # released nvidia/canary-qwen-2.5b checkpoint, so that IS the correct base.
    # S3-B3 (and any other full-decoder-FT run under salm_train_stable.py) was
    # never composed from the released checkpoint at all (random bridge init,
    # see ISS-013) -- loading its plain llm.model.layers.* keys onto the
    # released model's LoRA-shaped llm.base_model.model.model.layers.* keys
    # necessarily mismatches every LLM weight. 'composed' instead rebuilds the
    # exact architecture salm_train_stable.py trained (same _target_/
    # pretrained_llm/pretrained_asr as the run's own saved exp_config.yaml),
    # so the checkpoint's own key names match structurally.
    p.add_argument('--base', choices=['released', 'composed'], default='released')
    p.add_argument('--exp-config', default=None,
                    help="Path to the run's exp_config.yaml (written by salm_train_stable.py "
                         "next to its checkpoints). Required when --base composed.")
    args = p.parse_args()
    if args.base == 'composed' and not args.exp_config:
        p.error('--exp-config is required when --base composed')

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
    print(f'Loading model (base={args.base})...')
    if args.base == 'released':
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        base_desc = 'nvidia/canary-qwen-2.5b (released)'
    else:
        from omegaconf import OmegaConf
        exp_cfg = OmegaConf.load(args.exp_config)
        model = SALM(OmegaConf.to_container(exp_cfg.model, resolve=True))
        # Composed construction defaults to fp32 (no from_pretrained-style
        # dtype pass-through); the checkpoint's own tensors are fp16
        # (trained under precision: 16-true -- verified directly via
        # torch.load dtype inspection), and fp32 for this ~2.7B-param model
        # OOMs an 11GB 2080 Ti on its own weights alone. Cast to match.
        model = model.half()
        base_desc = (f"composed from {args.exp_config} "
                     f"(pretrained_llm={exp_cfg.model.pretrained_llm}, "
                     f"pretrained_asr={exp_cfg.model.pretrained_asr})")

    # ISS-013 (2026-09-10): strict=False was previously silently discarding
    # missing/unexpected keys. --base released's LLM keys are LoRA-shaped
    # (llm.base_model.model.model.layers.*); a non-LoRA full-decoder-FT
    # checkpoint (e.g. S3-B3) has plain llm.model.layers.* keys -- a
    # structural mismatch under which strict=False would let the ENTIRE LLM
    # silently fail to load, and this script would then report the BASE
    # model's WER as if it were the trained checkpoint's. Assert on the
    # actual incompatibility lists instead of discarding them, regardless of
    # which --base was used.
    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing keys, {n_unexpected} unexpected keys')
    if n_missing:
        print(f'  missing (first 10): {load_result.missing_keys[:10]}')
    if n_unexpected:
        print(f'  unexpected (first 10): {load_result.unexpected_keys[:10]}')
    # --base composed reconstructs the exact architecture the checkpoint was
    # trained under (same config, no LoRA wrapping) -- a healthy load has NO
    # excuse for any key mismatch, so require exactly 0 rather than a
    # tolerance. --base released keeps the small existing tolerance since
    # that path loads a checkpoint's weights onto an independently-constructed
    # LoRA-wrapped base which can have a handful of harmless buffer-naming
    # differences (audited 2026-09-10: unjustified as a blanket "<=5" magic
    # number, tightened for the path where 0 is actually the correct
    # expectation).
    max_allowed_mismatch = 0 if args.base == 'composed' else 5
    assert n_missing + n_unexpected <= max_allowed_mismatch, (
        f'ISS-013: {n_missing} missing + {n_unexpected} unexpected keys when loading '
        f'{args.checkpoint} onto {base_desc} -- this looks like a structural '
        f'mismatch (e.g. LoRA-shaped base model vs a plain full-parameter checkpoint), not '
        f'a handful of harmless buffer differences. Evaluating anyway would silently report '
        f"the BASE model's performance, not this checkpoint's. See research_log/ISSUES.md ISS-013."
    )
    model.cuda().eval()

    samples = [json.loads(l) for l in open(args.test_manifest)]
    if args.max_samples > 0: samples = samples[:args.max_samples]
    print(f'Evaluating {len(samples)} samples...')

    # Determinism fix (2026-09-10, ISS-013 follow-up): NOT passing a
    # generation_config here does NOT mean greedy decoding. transformers
    # backfills every DEFAULT-valued field of the config actually used from
    # the underlying LLM's own generation_config.json for any field this call
    # leaves untouched. Verified directly (audit): for --base released this
    # is harmless (canary-qwen-2.5b's LLM has pretrained_weights: false, so
    # its GenerationConfig is built via from_model_config and never reads a
    # generation_config.json -- do_sample stays False, matching every
    # historical "greedy" WER reported for v1/v2/v3). For --base composed,
    # the LLM is loaded via from_pretrained (pretrained_weights: true), which
    # DOES read Qwen3-1.7B's own generation_config.json
    # (do_sample=True, temperature=0.6, top_k=20, top_p=0.95) and backfills
    # all four for --base composed.
    #
    # First fix attempt passed generation_config=GenerationConfig(do_sample=False)
    # -- this does NOT work and was caught by a determinism check (running
    # the same 10 samples twice gave 38.10% then 36.19% WER). Root cause:
    # transformers' backfill checks whether a field's value EQUALS
    # GenerationConfig()'s own class-level default to decide whether to
    # override it from the model's generation_config.json -- and False IS
    # that class default for do_sample, so an explicit `do_sample=False` is
    # indistinguishable from "left unset" and gets overridden anyway.
    # Passing do_sample as a direct **kwarg to .generate() (not nested in a
    # GenerationConfig object) applies after that merge step and reliably
    # wins -- retested with a second determinism check (below, must match).
    gen_kwargs = dict(do_sample=False, num_beams=1, temperature=None, top_k=None, top_p=None)

    refs, hyps, errors = [], [], 0
    for i, s in enumerate(samples):
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(samples)} WER: {wer(refs, hyps)*100:.1f}%')
        try:
            ids = model.generate(prompts=[[{'role':'user',
                'content':f'Transcribe the following: {model.audio_locator_tag}',
                'audio':[s['audio_filepath']]}]], max_new_tokens=128,
                **gen_kwargs)
            refs.append(s['text'].lower().strip())
            hyps.append(model.tokenizer.ids_to_text(ids[0].cpu()).lower().strip())
        except Exception as e:
            # Audited 2026-09-10: a bare `except: errors += 1` silently
            # swallows every failure mode (OOM, audio-load fault, CUDA
            # error) with no visibility -- on a small sample count a
            # handful of hidden failures can invert a comparison. Print the
            # actual exception so a run with unexpectedly many errors can be
            # diagnosed instead of just counted.
            errors += 1
            print(f'  [error] sample {i} ({s.get("audio_filepath", "?")}): {type(e).__name__}: {e}')
            traceback.print_exc()

    final_wer = wer(refs, hyps)
    print(f'\nWER: {final_wer*100:.2f}% ({len(refs)} samples, {errors} errors)')
    json.dump({'wer': final_wer, 'samples': len(refs), 'errors': errors},
              open(args.output, 'w'), indent=2)

if __name__ == '__main__': main()
