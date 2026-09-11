# Engineering Memory Index

Initialized: 2026-09-06. All records below were recovered from existing repository files/git history (README.md, SUMMARY.md, REPLICATION_GUIDE.md, shared/*.md, models/*/docs/*.md, models/canary-qwen/docs/*.json, and `git show` on commits 34a5ca1 / 054bd54). No experiments were run and no historical documentation or results were modified to produce this index.

## Audits
- AUD-001 — ATCOSIM gender/speaker-independent split: train/test speaker overlap (leakage confirmed)
- AUD-002 — UWB-ATCC / ATCOSIM data split and format audit (as documented)
- AUD-003 — Documentation self-consistency corrections (WER metric naming, encoder-unfrozen param %)
- AUD-004 — Full repo structure/scripts/docs audit + hardware/software efficiency review (script convention split, stale 32.8% instance, SETUP.md gap, stale Phase 4 docs)
- AUD-005 — Independent fresh-eyes re-audit of Canary-Qwen v1/v2/v3 config rename (DEC-008): rename/fix confirmed correct, but found v1/v3 checkpoint-dir collision + stale REPLICATION_GUIDE.md rm -rf [see ISS-010]

## Validation
- VAL-001 — UWB-ATCC W2V2-large final checkpoint beats paper baseline (14.54%/12.69% vs paper 17.48-17.56%/13.72-14.26%)
- VAL-002 — ATCOSIM W2V2-large matches paper's 20k-step result (1.67%) in 5k steps — caveat: speaker overlap
- VAL-003 — Canary-Qwen v3 regularization breaks the ~24% WER plateau (confirms overfitting, not architecture)
- VAL-004 — EXP-007 female/male scripts confirmed torchrun/DDP, execution-identical to baseline
- VAL-005 — EXP-007 final preflight validation (manifests/config/output-safety/resource plan) — PASS
- VAL-006 — EXP-007 female run completion verification (real evidence) — PASS, WER=4.8468%
- VAL-007 — EXP-007 male run completion verification (real evidence) — PASS, WER=19.973%
- VAL-008 — Canary-Qwen N-best generation feasibility confirmed via source-code inspection (SALM.generate → HF GenerationConfig pass-through)
- VAL-009 — Spike D: full Canary-Qwen decoder fine-tuning fits in VRAM on 4×RTX 2080 Ti — GO, empirically confirmed (1-step smoke test)
- VAL-010 — Stage 2 fresh v1-equivalent control checkpoint trained + evaluated — WER=23.32%, matches historical figure exactly, now the live control for Stage 3
- VAL-011 — SUPERSEDED by VAL-012 — apparent "no surviving v3 model" conclusion was itself caused by a caching-bug artifact, not a real problem
- VAL-012 — FINAL: v1 (23.32%), v2 (23.82%), v3 (20.70%) all verified genuine via real inference, cross-checked against independent HuggingFace downloads and author's own result records — root cause of earlier confusion was a fixed-path caching collision in eval_finetuned.py
- VAL-013 — S4-FAIR: Canary-Qwen v1 + in-domain KenLM (N-best rescore) closes only ~1.5 WER points (23.32%->21.79%); fairness fix does not explain the W2V2-Canary gap
- VAL-014 — S4-FAIR: Canary-Qwen v3 + in-domain KenLM (N-best rescore) provides NO benefit (20.14% vs 19.42% beam-only, worse) — negative result, reported as-is
- VAL-015 — Gate 2 (500-step, production settings) PASS: first confirmed genuine gradient-driven learning after ISS-012's clip fix — val_loss 4.828→0.875 monotonic over real held-out data, 0.6% skip rate, checkpoint-verified real weight movement
- VAL-016 — Bridge-LR ablation: BRIDGE_LR=5e-4 confirmed necessary, not just plausible — uniform lr=1e-5 plateaus at val_loss 1.715 (2x worse than Gate 2's 0.875), both runs otherwise trained successfully
- VAL-017 — Gate 3 (2500-step budget) PASS: sustained genuine learning to val_loss 0.754 at step 843 (self-terminated via legitimate early stopping, not a bug), checkpoint-verified — strongest evidence yet the ISS-011/ISS-012 fixes hold at scale; open question whether early-stopping patience is too aggressive before the full production run

## Decisions
- DEC-001 — Discard ATCOSIM gender-based WER results; require a re-split for speaker independence
- DEC-002 — Canonicalize "greedy" vs "beam search" terminology and pick single canonical UWB-ATCC WER numbers
- DEC-003 — Use torchrun/DDP instead of paper's DataParallel launcher for W2V2-large training
- DEC-004 — Use FSDP (ModelParallelStrategy) instead of DDP for Canary-Qwen-2.5B training
- DEC-005 — Redesign research question around Canary-Qwen decoder-adaptation-scope study, drop "unseen domain"/architecture-vs-architecture framing (IEEE SLT review response)
- DEC-006 — Drop data-scale and cross-corpus studies from the P0/P1 execution-ready program (focus discipline)
- DEC-007 — Drop the ambiguous "research-optimized" (3e-5) Canary-Qwen ablation from the active manuscript/research record
- DEC-008 — Rename Canary-Qwen UWB-ATCC configs/scripts/results to canonical v1/v2/v3 scheme
- DEC-009 — Fix fp16 AdamW degeneracy with a custom fp32 master-weight optimizer, not bf16 (7.6x slower, measured) or 16-mixed (incompatible with ModelParallelStrategy)

## Experiments
- EXP-001 — UWB-ATCC W2V2-large fine-tuning (Phase 4) — 14.54%/12.69% WER [COMPLETE]
- EXP-002 — ATCOSIM W2V2-large fine-tuning (Phase 2) — 1.67%/1.28% WER [COMPLETE, caveat: leakage]
- EXP-003 — Canary-Qwen-2.5B zero-shot baseline on UWB-ATCC — 81.49% WER [COMPLETE]
- EXP-004 — Canary-Qwen-2.5B LoRA ablation series on UWB-ATCC (6 runs) — best v3 = 20.70% [COMPLETE]
- EXP-005 — Canary-Qwen-2.5B v1 (adapter-only) on ATCOSIM — 7.06% WER [COMPLETE]
- EXP-006 — Canary-Qwen-2.5B v3 (LoRA+SpecAugment) on ATCOSIM — 3.33% WER [COMPLETE]
- EXP-007 — ATCOSIM speaker-independent re-training (train_male/train_female) [COMPLETE 2026-09-07 — female WER=4.8468%, male WER=19.973%; large gender gap flagged as unexplained]
- EXP-008 — UWB-ATCC W2V2 dropout/mask_time_prob ablations [PROPOSED, not executed]
- EXP-009 — IEEE SLT review-response: Canary-Qwen decoder-adaptation-scope study [PROPOSED, not executed — see research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md; superseded by EXP-010's execution-ready version]
- EXP-010 — Execution-ready adaptive research program (staged, gated, GPU-hour-minimizing) [PROPOSED, not executed — see research_report/FINAL_RESEARCH_PROGRAM.md]
- EXP-011 — Re-run lower-LR (1e-4) Canary-Qwen UWB-ATCC ablation [PROPOSED, LOW PRIORITY, not executed — checkpoint lost, no citation-verified replacement yet]
- EXP-012 — S4-FAIR: decoding-fairness comparison (N-best+KenLM on Canary-Qwen v1) [COMPLETE 2026-09-08 — WER 23.32%->22.28% (beam)->21.79% (+KenLM); fairness fix does not explain the W2V2-Canary gap]
- EXP-013 — S4-FAIR: decoding-fairness comparison (N-best+KenLM on Canary-Qwen v3) [COMPLETE 2026-09-08 — WER 20.70%->19.42% (beam)->20.14% (+KenLM, WORSE); KenLM provides no benefit on the regularized checkpoint]

## Issues
- ISS-001 — ATCOSIM gender-subset evaluation has train/test speaker leakage [RESOLVED by discarding]
- ISS-002 — eval_model.py hypothesis-file bug when no LM is supplied [RESOLVED — documented workaround]
- ISS-003 — fp16 training NaN with small AdamW epsilon on Canary-Qwen FSDP [RESOLVED — eps=1e-4]
- ISS-004 — Stale trainable-param % (32.8 vs 29.2) in finetuned_results_v2.json [RESOLVED 2026-09-08]
- ISS-005 — Existing ATCOSIM 4-gram KenLM trained on leaked split; must not be used for speaker-independent decoding [OPEN]
- ISS-006 — ATCOSIM wrapper scripts lack `set -e`, silently report success after DDP ranks crash [RESOLVED — workaround, script not fixed]
- ISS-007 — Canary-Qwen v2 (encoder-unfrozen) config unresolved — WER=23.82% verified genuine via HF inference, but exact hyperparameters unrecoverable from any of 3 independent config sources checked [PARTIALLY RESOLVED]
- ISS-008 — No gender-stratified Canary-Qwen lhotse cuts exist yet for ATCOSIM speaker-independent evaluation [OPEN]
- ISS-009 — FULLY RESOLVED (2026-09-08): v3 checkpoint's config AND WER both verified genuine (20.70%, matches HF + author's own record). Earlier "doesn't reproduce" conclusion was a caching-bug artifact [see VAL-012]
- ISS-010 — v1/v3 share explicit_log_dir (framework-confirmed collision risk); train_canary_v1/v2.sh depend on an undocumented manual `cp` step not yet performed; REPLICATION_GUIDE.md §2.11 has a stale `rm -rf` that would delete v1+v3 checkpoints [OPEN]
- ISS-011 — CRITICAL: fp16 AdamW is numerically degenerate (SGD@lr/eps in disguise) in every Canary-Qwen run; caused S3-B3's divergence to inf; weight_decay has been inert in v1/v2/v3 (v3's gain is SpecAugment+dropout only, not weight_decay); tied-embedding untied by FSDP2 [OPEN, fix designed; Gate 1/2 validation runs superseded by ISS-012]
- ISS-012 — CRITICAL: gradient clipping silently zeroed 100% of gradients in Gate 1 v8/Gate 2 v2/Gate 2 v3 (fp16 DTensor norm-reduction overflow) — those runs took zero real optimizer steps; invalidates the flat-val_loss/bridge-LR finding and DEC-009's clip-value calibration [DIAGNOSED, fix designed]
- ISS-013 — HIGH: no Canary-Qwen run in this project (v1/v2/v3 or S3-B3) actually fine-tunes the released nvidia/canary-qwen-2.5b checkpoint (modality bridge always randomly re-initialized) — PARTIALLY RESOLVED: direct load-check confirms v1/v3's eval (0 missing/unexpected keys each) genuinely evaluated their own trained weights, not the released model; eval_finetuned.py now asserts on this for any future S3-B3 checkpoint [PARTIALLY RESOLVED]
- ISS-014 — CRITICAL: resuming corrupts the model — root cause CONFIRMED (torch.optim.Optimizer.load_state_dict silently downcasts fp32 master/exp_avg/exp_avg_sq to fp16, reproduced mechanistically at realistic gradient magnitudes) and FIXED (custom load_state_dict bypasses the downcast); validated via CPU repros + one real FSDP2 resume cycle (finite, correct). Second resume cycle blocked by [[ISS-015]] — do not resume production checkpoints until that's also resolved [FIXED for one resume cycle, blocked on ISS-015 for full validation]
- ISS-015 — Resuming under ModelParallelStrategy (multi-GPU, non-DDP) crashes at the SECOND post-resume validation inside Lightning's own EarlyStopping callback (cross-device best_score comparison) — unrelated to ISS-014, found while validating its fix [OPEN, not investigated]

## Environment
- ENV-001 — System hardware (4x RTX 2080 Ti, 11GB each, no sudo, Ubuntu 24)
- ENV-002 — W2V2 (idiap/w2v2-air-traffic) conda env `w2v2_asr` — versions and 8 known fixes
- ENV-003 — Canary-Qwen (NeMo speechlm2) conda env `canary_ft` — FSDP setup and known issues
- ENV-004 — No tmux/screen/sudo on this host; detached launches must explicitly activate conda env + PYTHONPATH
