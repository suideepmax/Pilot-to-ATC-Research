# Experiments

Provenance note (added 2026-09-07): Entries dated "recovered 2026-09-06" (EXP-001, EXP-002, EXP-007) were reconstructed by a prior Claude Code session from existing repository documentation and git history (see `research_log/INDEX.md`), not authored by the user. Each `Hypothesis:` field in those entries is that recovery session's synthesis of the project's documented intent — e.g., EXP-007's hypothesis was inferred from the leakage finding and Phase-4 TODO already described in `models/w2v2/docs/PROGRESS_ATCOSIM.md` and `SUMMARY.md` — not a verbatim quote from any file the user wrote. The underlying experimental design (train/test splits, what leakage was found) is accurately sourced from those docs; the specific hypothesis wording is not.

## EXP-001 — UWB-ATCC: wav2vec2-large-960h-lv60-self fine-tuning (Phase 4)

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Replicate the paper's (Zuluaga-Gomez et al.) UWB-ATCC W2V2-large fine-tuning result under local hardware constraints.

Hypothesis: A higher learning rate (5e-4) combined with DDP training can match or beat the paper's reported WER despite a smaller effective batch size than the paper used.

Dataset: UWB-ATCC, 80/20 split (seed=1234) — Train 11,543 utt / Test 2,886 utt (see [[AUD-002]]).

Dataset Version: as prepared by `models/w2v2/scripts/data_prepare_uwb_atcc.sh`.

Model: facebook/wav2vec2-large-960h-lv60-self (317M params, 100% fine-tuned, feature encoder frozen).

Configuration: steps=10,000; per_device_train_batch_size=1; gradient_accumulation=16; effective batch=64 (4 GPUs); lr=5e-4; warmup=1,000; mask_time_prob=0.01; fp16 enabled, fp16_full_eval disabled.

Random Seed: not explicitly recorded for the training run itself (data split seed=1234 is documented; training seed not found in repo).

Hardware: 4x NVIDIA RTX 2080 Ti (11GB each).

GPU Allocation: CUDA_VISIBLE_DEVICES=0,1,2,3, torchrun --nproc_per_node=4 (DDP; see [[DEC-003]]).

Command: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/uwb_atcc/train_w2v2_large-60v.sh` (per models/w2v2/docs/PROGRESS_UWB_ATCC.md Phase 4).

Estimated Cost: N/A (recovered post hoc).

Actual Runtime: Run 1: 7h34m, Run 2: 8h35m, Run 3: 7h15m (~7.8h average across 3 runs).

Results: 3 independent runs — Run 1: 15.17% WER (train loss 0.4062); Run 2: 15.15% WER (0.4076); Run 3: 15.07% WER (0.4077). Average 15.13% (training-time greedy eval). Standalone final eval (Phase 5, greedy, no LM): 14.54% (canonical, `finetuned_results_v2.json`); a second standalone eval of the same checkpoint gave 14.60% (documented run-to-run variation, [[DEC-002]]).

Metrics: WER (greedy, no LM) = 14.54%; WER (CTC + 4-gram KenLM) = 12.69%. Paper's published numbers: 17.48% (Table 3) / 17.56% (HuggingFace card) no-LM; 14.26% / 13.72% with LM.

Conclusion: The run beat the paper's reported WER on both no-LM and with-LM metrics. Attributed to the 5x higher learning rate (5e-4 vs. paper's 1e-4) combined with DDP producing cleaner logits (training-time greedy WER 15.07% here vs. paper's 29.81% at step 10k) that need less decoding correction (see [[DEC-002]], [[DEC-003]]).

Next Action: None outstanding for this experiment; superseded/refined numbers only affect labeling ([[DEC-002]]), not the underlying result.

Related Records: [[DEC-002]], [[DEC-003]], [[VAL-001]], [[ENV-002]]

---

## EXP-002 — ATCOSIM: wav2vec2-large-960h-lv60-self fine-tuning (Phase 2)

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Fine-tune the same W2V2-large model on ATCOSIM and compare against the paper's published ATCOSIM model.

Hypothesis: The paper's reported 1.67% WER (achieved at 20,000 steps) can be matched in fewer steps due to a smaller effective batch size producing more gradient updates per epoch.

Dataset: ATCOSIM, 80/20 split (seed=1234) — Train 7,660 utt / Test 1,916 utt (see [[AUD-002]]).

Model: facebook/wav2vec2-large-960h-lv60-self (317M params, 100% fine-tuned).

Configuration: steps=5,000; per_device_train_batch_size=1; gradient_accumulation=16; effective batch=64 (paper: batch 24, grad_acc 4, effective 96, on 1 GPU); lr=5e-4; mask_time_prob=0.01; min_duration_in_seconds raised to 0.5 (from paper's 0.2 — shorter clips crashed the time-mask, seq_len < mask_len=12); fp16_full_eval disabled (CUBLAS crash workaround on RTX 2080 Ti).

Hardware: 4x NVIDIA RTX 2080 Ti; DDP via torchrun (paper used 1 GPU DataParallel) — see [[DEC-003]].

Command: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash models/w2v2/scripts/train_wav2vec2_atcosim_large.sh`.

Actual Runtime: ~3.8 hours.

Results (learning curve, greedy eval WER): step 500=4.13%, 1000=3.48%, 2000=2.14%, 3000=1.97%, 4000=1.79%, 4500=1.66% (best), 5000=1.67% (final). Train loss at step 5000: 0.0216.

Metrics: WER (greedy, no LM, step 5000) = 1.67% — exactly matches paper's 20,000-step result (paper reached 2.10% at their own step 5,000/epoch 64; 1.67% at step 20,000/epoch 256). WER with CTC+4-gram KenLM = 1.28% (Phase 3).

Conclusion: Matched the paper's final WER (1.67%) in 1/4 the training steps, attributed to smaller effective batch (64 vs. 96) giving more gradient updates per epoch (42 epochs here at step 5000 vs. paper's 64 epochs at their step 5000). IMPORTANT CAVEAT: this result is confounded by train/test speaker overlap in the random 80/20 split (all 10 speakers appear in both sets) and ~42 epochs of training on a small corpus — the 1.67% figure is not a speaker-independent or difficulty-matched comparison to UWB-ATCC's 14.54% (see PROGRESS_ATCOSIM.md "Notes" and SUMMARY.md "Why ATCOSIM WER is so much lower than UWB-ATCC").

Next Action: A speaker-independent re-run (train_male/train_female vs. held-out speakers) is proposed but not yet executed — see [[DEC-001]], [[EXP-007]] (proposed).

Related Records: [[DEC-001]], [[DEC-003]], [[ISS-001]], [[VAL-002]]

---

## EXP-003 — Canary-Qwen-2.5B zero-shot baseline on UWB-ATCC

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Establish a zero-shot (no fine-tuning) WER baseline for nvidia/canary-qwen-2.5b on UWB-ATCC before any fine-tuning experiments.

Dataset: UWB-ATCC test split, 2,886 utterances, resampled to 16kHz.

Model: nvidia/canary-qwen-2.5b, no fine-tuning.

Results: WER = 81.49% (`models/canary-qwen/docs/baseline_results.json`: wer=0.8149139054675173, num_samples=2886, errors=0).

Conclusion: Confirms the model requires domain fine-tuning for ATC speech; used as the step-0 anchor point in all Canary learning curves.

Related Records: [[EXP-004]]

---

## EXP-004 — Canary-Qwen-2.5B LoRA fine-tuning ablation series on UWB-ATCC

Date: recovered 2026-09-06
Status: COMPLETE (6 documented runs)

Objective: Find a Canary-Qwen fine-tuning configuration on UWB-ATCC that closes the gap to W2V2's 14.54% WER, starting from an initial LoRA config that plateaued near 24%.

Dataset: UWB-ATCC, same split as EXP-001 (Train 11,543 / Test 2,886), audio resampled 8kHz→16kHz.

Model: nvidia/canary-qwen-2.5b (2.87B total params), SALM architecture (FastConformer encoder + Qwen3-1.7B LLM), fine-tuned via NeMo speechlm2. Framework/strategy: FSDP (ModelParallelStrategy), precision 16-true, AdamW eps=1e-4 (see [[DEC-004]], [[ISS-003]]).

Hardware: 4x RTX 2080 Ti.

Runs (all 10,000 steps except where noted), per models/canary-qwen/docs/PROGRESS.md:

| Run | Config | Trainable params | WER | val_loss best |
|---|---|---|---|---|
| 1 — Original LoRA | lr=5e-4, warmup=1000, dropout=0.01, no SpecAugment, WD=1e-3 | 27.8M (0.97%) | 23.32% | 0.678 |
| 2 — Encoder Unfrozen | same as Run 1 + FastConformer encoder unfrozen | 838.8M (29.2%; corrected from 32.8%, see [[AUD-003]]) | 23.82% | 0.649 |
| 3 — Lower LR | lr=1e-4 | 27.8M | 32.58% | 0.762 (overfits after step 2500, val_loss rises to 1.109 by 10k) |
| 4 — Research-optimized (r=64) | lr=3e-5, r=64, 2500 steps | n/a | FAILED (NaN at step 1500, eps=1e-6 too small; also r=64 vs pretrained r=128 mismatch prevented eval) | — |
| 5 — Research-optimized v2 (r=128) | lr=3e-5, r=128, 2500 steps | 27.8M | 60.46% | 1.419 (still decreasing — too few steps at this LR) |
| 6 — v3 (best) | lr=5e-4, warmup=1000, dropout=0.1, SpecAugment ON (2 freq masks, 10 time masks), WD=1e-2, LoRA r=128/alpha=256 on q_proj+v_proj | 27.8M (0.97%) | **20.70%** | 0.581 |

Metrics (learning curve at step 10,000 unless noted, 500-sample eval subset): Run 1 (orig): 24.53%; Run 2 (unfrozen): 23.89%; Run 6 (v3): 22.30% (500-sample); full test-set (2,886 samples) v3 = 20.70% (`v3_results.json`).

Conclusion: The ~24% WER plateau (Runs 1-2) was caused by overfitting on the small (10.5h) training set, not by the frozen LLM decoder being an architectural bottleneck — Run 6 (v3) broke through to 20.70% purely via added regularization (SpecAugment + 10x dropout + 10x weight decay) at the same LR. Unfreezing the encoder (29.2% of params) without regularization did not beat LoRA alone (0.97% of params): 23.82% vs 23.32%. Learning rate must stay high (5e-4); both lower-LR variants (Runs 3, 5) underperformed.

Next Action: None outstanding — v3 is the adopted best config for UWB-ATCC and was reused as-is for the ATCOSIM v3 run (see [[EXP-006]]).

Related Records: [[ISS-003]], [[DEC-004]], [[AUD-003]], [[VAL-003]]

---

## EXP-005 — Canary-Qwen-2.5B v1 (adapter-only) fine-tuning on ATCOSIM

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Establish an initial Canary-Qwen fine-tuning result on ATCOSIM before applying the UWB-ATCC v3 regularization recipe.

Dataset: ATCOSIM, Lhotse CutSet JSONL derived from Kaldi format, audio resampled 32kHz→16kHz.

Model: nvidia/canary-qwen-2.5b, modality-adapter-only fine-tuning (no LoRA).

Configuration: trainable params = 2.1M (0.07%); max_steps=5,000; limit_train_batches=500; batch_size=1; accumulate_grad_batches=4; effective data epochs ≈2.6; precision bf16-true.

Results: best val_loss = 0.17676 at step 3500. WER = 7.06% (best checkpoint, step 3500).

Conclusion: With the LLM fully frozen and only a small modality-adapter bridge trained, WER of 7.06% was achieved — substantially worse than W2V2's 1.67%/1.28% on the same corpus, motivating the LoRA + SpecAugment follow-up (EXP-006).

Related Records: [[EXP-006]]

---

## EXP-006 — Canary-Qwen-2.5B v3 (LoRA + SpecAugment) fine-tuning on ATCOSIM

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Apply the UWB-ATCC v3 regularized-LoRA recipe (EXP-004 Run 6) to ATCOSIM to see if it closes the gap to W2V2.

Dataset: ATCOSIM, same as EXP-005.

Model: nvidia/canary-qwen-2.5b, LoRA r=128/alpha=256/dropout=0.1 on q_proj+v_proj + modality adapter; SpecAugment (2 freq masks, 10 time masks).

Configuration: trainable params = 27.8M (0.97%); max_steps=10,000; limit_train_batches=1,000; batch_size=2; accumulate_grad_batches=4; effective data epochs ≈10.4; weight_decay=1e-2; warmup=1,000; precision 16-true; prompt = "Transcribe the following:".

Results: best val_loss = 0.07899 at step 7000. WER = 3.33% (`v3_results.json` context: this file is under canary-qwen/docs but the 3.33% figure for ATCOSIM is reported in PROGRESS_ATCOSIM.md and README.md/SUMMARY.md; note `v3_results.json` in the docs directory actually stores 0.2070... which is the UWB-ATCC v3 result — the ATCOSIM v3 WER of 3.33% was not found stored in its own JSON artifact in this repo, only in Markdown docs).

Metrics: v1→v3 improvement: 7.06% → 3.33% (roughly 2x). W2V2 comparison on same corpus: 1.67% (no LM) / 1.28% (with KenLM) still leads.

Conclusion: LoRA r=128 (27.8M trainable vs. v1's 2.1M) plus SpecAugment plus greater data coverage (~10.4 vs ~2.6 effective epochs) roughly halved WER. W2V2 still leads on ATCOSIM, attributed to ATCOSIM being clean/scripted/close-talk audio where full CTC fine-tuning with many epochs dominates; the Canary gap is expected to close further on noisier corpora like UWB-ATCC where LM-style priors matter more (per PROGRESS_ATCOSIM.md discussion).

Limitations: The 3.33% WER figure could not be traced to a JSON evidence artifact in this repo (only Markdown docs) — flagged, not fabricated; treat as reported-but-not-independently-file-verified within this repo snapshot.

Related Records: [[EXP-004]], [[EXP-005]], [[ISS-004]]

---

## EXP-007 — ATCOSIM speaker-independent re-training (train_male/train_female vs. held-out speakers)

Date: recovered 2026-09-06 (proposed); both runs executed and completed 2026-09-07
Status: COMPLETE

Objective: Produce a valid speaker-independent ATCOSIM WER by training separate models on `train_male`/`train_female` splits and evaluating only against the disjoint held-out speakers (test_male/test_female), replacing the leaked gender-eval results from [[ISS-001]].

Hypothesis: WER on truly held-out speakers will be materially higher than the leaked 0.01%/0.86% figures, giving a fair generalization estimate for ATCOSIM. (Provenance: this hypothesis sentence was synthesized by the 2026-09-06 recovery session from PROGRESS_ATCOSIM.md's leakage finding — it is not a verbatim quote from project documentation. See the provenance note at the top of this file.)

Dataset: ATCOSIM gender subsets — train_female={zf1,zf2,gf1}/test_female={zf3}; train_male={sm1,sm2,sm3,sm4}/test_male={gm1,gm2} (data already exists per shared/data_info.md; not yet used for a leakage-free training run).

Status note: Listed as pending in SUMMARY.md "Pending / Next Steps" under "ATCOSIM Phase 4 (valid)" and PROGRESS_ATCOSIM.md Phase 4 as "That is a different experiment (Phase 4 remains TODO)." No training run, checkpoint, or result exists in the repo for this experiment as of the latest commit inspected (054bd54).

Preflight update (2026-09-07): Final preflight validated — GPU/DDP execution model confirmed equivalent to baseline ([[VAL-004]]), and manifests/config/output-safety/resource-plan checked with no scientific or data blocker ([[VAL-005]]). One new risk was discovered and documented, not yet mitigated: the existing ATCOSIM 4-gram KenLM binary was trained on the leaked main split and must not be used for this experiment's decoding ([[ISS-005]]). No male or female smoke test has actually been executed in this repository — only static config/manifest inspection has occurred at preflight time. Status remains PROPOSED pending explicit user approval to launch production training.

Launch update (2026-09-07): User approved production launch. Female run started via detached `setsid`/`nohup` (tmux unavailable on this host, see [[ENV-004]]). First launch attempt failed silently (all 4 DDP ranks crashed on `ModuleNotFoundError: No module named 'datasets'` due to wrong conda env in the detached shell; wrapper scripts lack `set -e` and printed false "Done"/"Training complete" messages anyway — see [[ISS-006]]). Relaunched with `w2v2_asr` conda env explicitly activated and `PYTHONPATH` set; confirmed via `ps`/`nvidia-smi` that all 4 DDP rank processes were alive, one per GPU, with real training progress.

Female result (2026-09-07, verified — see [[VAL-006]]): **eval_wer = 4.8468%**, eval_loss = 0.0857, train_loss = 0.3925, train_runtime = 7006.3s (≈1.95h), final epoch 47.17 (max_steps=2500 reached), evaluated on the truly held-out `test_female` (zf3) speaker. This is the legitimate speaker-independent female result replacing ISS-001's leaked 0.86% figure — WER is materially higher than the leaked figure, consistent with the experiment's hypothesis. Checkpoint and results artifacts at `experiments/results/speaker_independent_female/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc/`.

Male run launched 2026-09-07 immediately after female verification, using the corrected launch invocation from the start (conda `w2v2_asr` + `PYTHONPATH=` in the detached shell) — succeeded on the first attempt, no repeat of the ISS-006/ENV-004 failure. 4 DDP rank processes confirmed via `ps`/`nvidia-smi` (GPUs 1-3 at 100% util).

Male result (2026-09-07, verified — see [[VAL-007]]): **eval_wer = 19.973%**, eval_loss = 0.5306, train_loss = 0.4119, train_runtime = 8623.6s (≈2.40h), final epoch 39.99 (max_steps=3000 reached), evaluated on the truly held-out `test_male` (gm1, gm2) speakers. This is the legitimate speaker-independent male result replacing ISS-001's leaked 0.01% figure. Checkpoint and results artifacts at `experiments/results/speaker_independent_male/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc/`.

Conclusion: EXP-007's hypothesis is confirmed — WER on truly held-out speakers is materially higher than the leaked figures in both genders (female: 4.8468% vs. leaked 0.86%; male: 19.973% vs. leaked 0.01%). The large female/male gap (4.85% vs. 19.97%) is itself a new, unexplained finding — not predicted by the original hypothesis, which only concerned the direction of the leak correction, not a male/female asymmetry. Candidate explanations not yet investigated: relative train-set size (train_male=4,849 vs. train_female=3,471 utterances — male has *more* training data yet performs far worse, so size alone does not explain it), speaker count per split (4 male vs. 3 female training speakers), or accent/individual-speaker variability among the ATCOSIM non-native speakers. This asymmetry should be investigated before citing a single "ATCOSIM speaker-independent WER" figure, since female and male differ by roughly 4x.

Next Action: Investigate the female/male WER asymmetry (candidate factors above) before treating either number as "the" ATCOSIM speaker-independent WER in cross-paper or cross-model comparisons. No further training is required to answer this — start with per-speaker WER breakdown on the existing eval sets and a check of individual test-male speaker (gm1 vs gm2) difficulty, before proposing a new experiment.

Related Records: [[ISS-001]], [[DEC-001]], [[VAL-004]], [[VAL-005]], [[VAL-006]], [[VAL-007]], [[ISS-005]], [[ISS-006]], [[ENV-004]]

---

## EXP-008 — UWB-ATCC W2V2 hyperparameter ablations (dropout, mask_time_prob)

Date: recovered 2026-09-06 (proposed; not yet executed)
Status: PROPOSED

Objective: Study the effect of dropout and mask_time_prob on UWB-ATCC W2V2 fine-tuning WER.

Evidence of proposal only: SUMMARY.md "Pending / Next Steps" — "UWB-ATCC W2V2 ablations: Effect of dropout, mask_time_prob on UWB-ATCC W2V2." No configuration, run, or result exists in the repo.

Related Records: [[EXP-001]]
