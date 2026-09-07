# Engineering Memory Index

Initialized: 2026-09-06. All records below were recovered from existing repository files/git history (README.md, SUMMARY.md, REPLICATION_GUIDE.md, shared/*.md, models/*/docs/*.md, models/canary-qwen/docs/*.json, and `git show` on commits 34a5ca1 / 054bd54). No experiments were run and no historical documentation or results were modified to produce this index.

## Audits
- AUD-001 — ATCOSIM gender/speaker-independent split: train/test speaker overlap (leakage confirmed)
- AUD-002 — UWB-ATCC / ATCOSIM data split and format audit (as documented)
- AUD-003 — Documentation self-consistency corrections (WER metric naming, encoder-unfrozen param %)

## Validation
- VAL-001 — UWB-ATCC W2V2-large final checkpoint beats paper baseline (14.54%/12.69% vs paper 17.48-17.56%/13.72-14.26%)
- VAL-002 — ATCOSIM W2V2-large matches paper's 20k-step result (1.67%) in 5k steps — caveat: speaker overlap
- VAL-003 — Canary-Qwen v3 regularization breaks the ~24% WER plateau (confirms overfitting, not architecture)
- VAL-004 — EXP-007 female/male scripts confirmed torchrun/DDP, execution-identical to baseline
- VAL-005 — EXP-007 final preflight validation (manifests/config/output-safety/resource plan) — PASS
- VAL-006 — EXP-007 female run completion verification (real evidence) — PASS, WER=4.8468%
- VAL-007 — EXP-007 male run completion verification (real evidence) — PASS, WER=19.973%

## Decisions
- DEC-001 — Discard ATCOSIM gender-based WER results; require a re-split for speaker independence
- DEC-002 — Canonicalize "greedy" vs "beam search" terminology and pick single canonical UWB-ATCC WER numbers
- DEC-003 — Use torchrun/DDP instead of paper's DataParallel launcher for W2V2-large training
- DEC-004 — Use FSDP (ModelParallelStrategy) instead of DDP for Canary-Qwen-2.5B training

## Experiments
- EXP-001 — UWB-ATCC W2V2-large fine-tuning (Phase 4) — 14.54%/12.69% WER [COMPLETE]
- EXP-002 — ATCOSIM W2V2-large fine-tuning (Phase 2) — 1.67%/1.28% WER [COMPLETE, caveat: leakage]
- EXP-003 — Canary-Qwen-2.5B zero-shot baseline on UWB-ATCC — 81.49% WER [COMPLETE]
- EXP-004 — Canary-Qwen-2.5B LoRA ablation series on UWB-ATCC (6 runs) — best v3 = 20.70% [COMPLETE]
- EXP-005 — Canary-Qwen-2.5B v1 (adapter-only) on ATCOSIM — 7.06% WER [COMPLETE]
- EXP-006 — Canary-Qwen-2.5B v3 (LoRA+SpecAugment) on ATCOSIM — 3.33% WER [COMPLETE]
- EXP-007 — ATCOSIM speaker-independent re-training (train_male/train_female) [COMPLETE 2026-09-07 — female WER=4.8468%, male WER=19.973%; large gender gap flagged as unexplained]
- EXP-008 — UWB-ATCC W2V2 dropout/mask_time_prob ablations [PROPOSED, not executed]

## Issues
- ISS-001 — ATCOSIM gender-subset evaluation has train/test speaker leakage [RESOLVED by discarding]
- ISS-002 — eval_model.py hypothesis-file bug when no LM is supplied [RESOLVED — documented workaround]
- ISS-003 — fp16 training NaN with small AdamW epsilon on Canary-Qwen FSDP [RESOLVED — eps=1e-4]
- ISS-004 — Stale trainable-param % (32.8 vs 29.2) in finetuned_results_unfrozen.json [OPEN]
- ISS-005 — Existing ATCOSIM 4-gram KenLM trained on leaked split; must not be used for speaker-independent decoding [OPEN]
- ISS-006 — ATCOSIM wrapper scripts lack `set -e`, silently report success after DDP ranks crash [RESOLVED — workaround, script not fixed]

## Environment
- ENV-001 — System hardware (4x RTX 2080 Ti, 11GB each, no sudo, Ubuntu 24)
- ENV-002 — W2V2 (idiap/w2v2-air-traffic) conda env `w2v2_asr` — versions and 8 known fixes
- ENV-003 — Canary-Qwen (NeMo speechlm2) conda env `canary_ft` — FSDP setup and known issues
- ENV-004 — No tmux/screen/sudo on this host; detached launches must explicitly activate conda env + PYTHONPATH
