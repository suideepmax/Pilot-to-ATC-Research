#!/usr/bin/env python3
"""Teacher-forced audio-grounding diagnostic (2026-09-10, per systems-architect
audit recommendation: cheaper and more decisive than a generation-based
audio-mismatch test -- no sampling noise, no variable-length-prompt confound,
fully deterministic).

Loads one real validation batch (same data pipeline the training run itself
uses, not a hand-rolled substitute), computes the model's own training-time
cross-entropy loss twice: once with the batch's real audio-to-transcript
pairing, once with the audio tensor permuted within the batch (a derangement,
no fixed points) while text/loss_mask stay untouched. If the model's loss is
materially insensitive to which audio it receives, its transcription-loss
improvements are not (or not only) coming from genuine acoustic grounding.

This is inference-only: no gradient step, no optimizer, no resume. Does not
touch ISS-014/ISS-015 at all.

Usage:
    CUDA_VISIBLE_DEVICES=1 python audio_grounding_check.py \
        --exp-config ~/canary-ft/experiments_s3b3_gate3/exp_config.yaml \
        --checkpoint ~/canary-ft/experiments_s3b3_gate3/checkpoints/step=843-last.ckpt \
        --batch-size 16
"""
import argparse
import hashlib
import os

import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def derangement(n: int, seed: int = 0) -> list[int]:
    """A permutation of range(n) with no fixed points (no i maps to i)."""
    import random
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def compute_loss(model, batch):
    inputs = model.prepare_inputs(batch)
    with torch.no_grad():
        forward_outputs = model(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
    num_frames = (inputs["target_ids"] != -100).long().sum()
    loss = torch.nn.functional.cross_entropy(
        forward_outputs["logits"].flatten(0, 1),
        inputs["target_ids"].flatten(0, 1),
        reduction="sum",
        ignore_index=-100,
    ) / num_frames
    preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
    refs = inputs["target_ids"].reshape(-1)
    preds = preds[refs != -100]
    refs = refs[refs != -100]
    acc = preds.eq(refs).float().mean()
    return loss.item(), acc.item(), int(num_frames.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-batches', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    ckpt_hash = hashlib.sha256(os.path.abspath(args.checkpoint).encode()).hexdigest()[:16]
    consolidated = f'/tmp/canary_eval_consolidated_{ckpt_hash}.pt'
    if not os.path.exists(consolidated):
        print(f'Consolidating {args.checkpoint} -> {consolidated}...')
        dcp_to_torch_save(args.checkpoint, consolidated)
    state = torch.load(consolidated, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']

    from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset

    exp_cfg = OmegaConf.load(args.exp_config)
    model = SALM(OmegaConf.to_container(exp_cfg.model, resolve=True))
    model = model.half()
    load_result = model.load_state_dict(state, strict=False)
    n_missing, n_unexpected = len(load_result.missing_keys), len(load_result.unexpected_keys)
    print(f'load_state_dict: {n_missing} missing, {n_unexpected} unexpected keys')
    assert n_missing == 0 and n_unexpected == 0, (
        f'Structural mismatch loading {args.checkpoint} -- refusing to run a diagnostic '
        f'against a model that is not actually this checkpoint. missing={load_result.missing_keys[:5]} '
        f'unexpected={load_result.unexpected_keys[:5]}'
    )
    model = model.cuda().eval()

    data_cfg = OmegaConf.to_container(exp_cfg.data, resolve=True)
    # Larger batch size than training's batch_size=1 so there is something to
    # permute within a batch; deterministic (shard_seed already fixed in the
    # dev config, shuffle left as configured for validation_ds -- normally
    # false already).
    data_cfg['validation_ds']['batch_size'] = args.batch_size
    dm = DataModule(OmegaConf.create(data_cfg), tokenizer=model.tokenizer, dataset=SALMDataset(tokenizer=model.tokenizer))
    val_loader = dm.val_dataloader()

    print(f'Running {args.num_batches} batch(es) of size {args.batch_size}...')
    correct_losses, correct_accs = [], []
    perm_losses, perm_accs = [], []
    n_seen = 0
    for batch_idx, (combined_batch, _, _) in enumerate(val_loader):
        if n_seen >= args.num_batches:
            break
        for name, batch in combined_batch.items():
            if batch is None or n_seen >= args.num_batches:
                continue
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items() if k != 'conversations'}
            B = batch['audios'].shape[0]
            if B < 2:
                print(f'  batch {batch_idx} ({name}): B={B} < 2, cannot derange, skipping')
                continue

            loss_c, acc_c, nframes = compute_loss(model, batch)

            perm = derangement(B, seed=args.seed + batch_idx)
            perm_batch = dict(batch)
            perm_batch['audios'] = batch['audios'][perm]
            perm_batch['audio_lens'] = batch['audio_lens'][perm]
            loss_p, acc_p, _ = compute_loss(model, perm_batch)

            print(f'  batch {batch_idx} ({name}): B={B} num_frames={nframes} | '
                  f'correct-audio loss={loss_c:.4f} acc={acc_c:.4f} | '
                  f'permuted-audio loss={loss_p:.4f} acc={acc_p:.4f} | '
                  f'delta(loss)={loss_p - loss_c:+.4f} delta(acc)={acc_p - acc_c:+.4f}')
            correct_losses.append(loss_c); correct_accs.append(acc_c)
            perm_losses.append(loss_p); perm_accs.append(acc_p)
            n_seen += 1

    if not correct_losses:
        print('No usable batches (all had B<2 or were exhausted) -- cannot report.')
        return

    import statistics
    print()
    print(f'Mean correct-audio loss:   {statistics.mean(correct_losses):.4f} (n={len(correct_losses)})')
    print(f'Mean permuted-audio loss:  {statistics.mean(perm_losses):.4f}')
    print(f'Mean correct-audio acc:    {statistics.mean(correct_accs):.4f}')
    print(f'Mean permuted-audio acc:   {statistics.mean(perm_accs):.4f}')
    print(f'Mean delta(loss), permuted-correct: {statistics.mean(p - c for p, c in zip(perm_losses, correct_losses)):+.4f}')
    print(f'Mean delta(acc), permuted-correct:  {statistics.mean(p - c for p, c in zip(perm_accs, correct_accs)):+.4f}')


if __name__ == '__main__':
    main()
