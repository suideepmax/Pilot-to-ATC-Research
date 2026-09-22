# Experiments

Provenance note (added 2026-09-07): Entries dated "recovered 2026-09-06" (EXP-001, EXP-002, EXP-007) were reconstructed from existing repository documentation and git history (see `research_log/INDEX.md`), not authored by the user. Each `Hypothesis:` field in those entries is a synthesis of the project's documented intent — e.g., EXP-007's hypothesis was inferred from the leakage finding and Phase-4 TODO already described in `models/w2v2/docs/PROGRESS_ATCOSIM.md` and `SUMMARY.md` — not a verbatim quote from any file the user wrote. The underlying experimental design (train/test splits, what leakage was found) is accurately sourced from those docs; the specific hypothesis wording is not.

Model-provenance note (added 2026-09-10, [[ISS-013]]): every Canary-Qwen fine-tuning run below (EXP-004 through S3-B3) composes its model fresh from `pretrained_llm: Qwen/Qwen3-1.7B` + `pretrained_asr: nvidia/canary-1b-flash` (verified directly in each run's own `.yaml`, e.g. `salm_uwb_atcc_v1.yaml`'s `model:` block) -- NOT via `SALM.from_pretrained('nvidia/canary-qwen-2.5b')`. The modality-bridge layer (`perception.proj`) is therefore randomly initialized at the start of every one of these runs, never loaded from the released checkpoint's own trained bridge. Below, "Model: nvidia/canary-qwen-2.5b" names the architecture being replicated (same LLM+encoder+bridge composition, same published recipe), not a claim that the released checkpoint's weights were loaded and fine-tuned in place. The one exception is EXP-003 (zero-shot baseline), which genuinely IS `SALM.from_pretrained('nvidia/canary-qwen-2.5b')` unmodified -- no fine-tuning, no composition.

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

**RENAMED (2026-09-08, [[DEC-008]]):** runs below are now referred to by the canonical v1/v2/v3 scheme (matching `salm_uwb_atcc_v1/v2/v3.yaml`, `train_canary_v1/v2/v3.sh`), not the original "Run N" numbering. All three WER numbers were independently re-verified via real inference this session — see [[VAL-012]]. The "research-optimized" runs (formerly Run 4/5) are **dropped** from the active record per [[DEC-007]] — retained below only for historical completeness, not as part of the canonical comparison.

Runs (all 10,000 steps except where noted), per models/canary-qwen/docs/PROGRESS.md:

| Run | Config | Trainable params | WER | val_loss best |
|---|---|---|---|---|
| **v1 — LoRA baseline** (formerly "Run 1 — Original LoRA") | lr=5e-4, warmup=1000, dropout=0.01, no SpecAugment, WD=1e-3 | 27.8M (0.97%) | 23.32% (re-verified [[VAL-010]]) | 0.678 |
| **v2 — Encoder Unfrozen** (formerly "Run 2") | same as v1 + FastConformer encoder unfrozen (**config to reproduce this is unresolved — see [[ISS-007]]**) | 838.8M (29.2%; corrected from 32.8%, see [[AUD-003]]) | 23.82% (re-verified via HF inference, [[VAL-012]]) | 0.649 |
| Lower LR (not part of v1/v2/v3; checkpoint lost, planned re-run [[EXP-011]]) | lr=1e-4 | 27.8M | 32.58% (citation only, not re-verified) | 0.762 (overfits after step 2500, val_loss rises to 1.109 by 10k) |
| ~~Research-optimized (r=64)~~ — **DROPPED, [[DEC-007]]** | lr=3e-5, r=64, 2500 steps | n/a | FAILED (NaN at step 1500, eps=1e-6 too small; also r=64 vs pretrained r=128 mismatch prevented eval) | — |
| ~~Research-optimized v2 (r=128)~~ — **DROPPED, [[DEC-007]]** | lr=3e-5, r=128, 2500 steps | 27.8M | 60.46% | 1.419 (still decreasing — too few steps at this LR) |
| **v3 — LoRA + Regularization (best)** (formerly "Run 6") | lr=5e-4, warmup=1000, dropout=0.1, SpecAugment ON (2 freq masks, 10 time masks), WD=1e-2, LoRA r=128/alpha=256 on q_proj+v_proj | 27.8M (0.97%) | **20.70%** (re-verified twice, [[VAL-012]]) | 0.581 |

Metrics (learning curve at step 10,000 unless noted, 500-sample eval subset): v1: 24.53%; v2 (unfrozen): 23.89%; v3: 22.30% (500-sample); full test-set (2,886 samples) v3 = 20.70% (`finetuned_results_v3.json`, formerly `v3_results.json`).

Conclusion: The ~24% WER plateau (v1/v2) was caused by overfitting on the small (10.5h) training set, not by the frozen LLM decoder being an architectural bottleneck — v3 broke through to 20.70% via added regularization at the same LR. **CORRECTED (2026-09-10, [[ISS-011]]):** attributing this to "10x weight decay" was wrong -- fp16 training under plain `torch.optim.AdamW` (used for v1/v2/v3, before [[DEC-009]]'s fp32-master-weight fix) makes `exp_avg_sq` underflow to exactly zero at this model's real gradient magnitudes, which degenerates AdamW's update into `-lr * sign(grad)`-style SGD and makes the `weight_decay` term numerically inert regardless of its configured value -- v3's gain is attributable to SpecAugment + 10x dropout only, not weight decay. Unfreezing the encoder (29.2% of params, v2) without regularization did not beat LoRA alone (0.97% of params, v1): 23.82% vs 23.32%. Learning rate must stay high (5e-4); the lower-LR variant underperformed (dropped variants excluded from this conclusion per [[DEC-007]]).

Next Action: v3 is the adopted best config for UWB-ATCC and was reused as-is for the ATCOSIM v3 run (see [[EXP-006]]). Outstanding: v2's exact reproducing config remains unresolved ([[ISS-007]]) — worth a fresh, correctly-configured retrain if a config-verified v2 result is needed (not yet scheduled, see this session's discussion). Lower-LR retrain planned at low priority ([[EXP-011]]).

Related Records: [[ISS-003]], [[ISS-007]], [[DEC-004]], [[DEC-007]], [[DEC-008]], [[AUD-003]], [[VAL-003]], [[VAL-012]]

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

## EXP-009 — IEEE SLT review-response program: Canary-Qwen decoder-adaptation-scope study

Date: 2026-09-07 (proposed; not yet executed)
Status: PROPOSED

Objective: Execute the P0 experiment program from `research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` and [[DEC-005]] — a controlled study of Canary-Qwen decoder-adaptation scope (LoRA q/v-only baseline → broader-target LoRA → full decoder fine-tune, gated on a VRAM feasibility check) paired with a sequential component-wise regularization ablation and a decoding-fairness-corrected comparison against the W2V2-large CTC baseline.

Hypothesis: Broadening Canary-Qwen's decoder adaptation scope beyond q/v-only LoRA will measurably reduce WER on UWB-ATCC; the magnitude of that reduction determines whether the redesigned paper's headline claim is "adaptation scope explains most of the gap to W2V2" or "adaptation scope alone does not explain the gap, implicating architecture/pretraining more fundamentally."

Dataset: UWB-ATCC (train=10.54h/11,543 utt, test=2.63h/2,886 utt, verified this session via direct `segments` duration computation).

Model: nvidia/canary-qwen-2.5b (SALM), same base as prior runs.

Configuration: Staged per the report's Section 16 roadmap — Stage 0 feasibility spikes (N-best generation support in installed NeMo; full-decoder-FT VRAM fit), Stage 1 smoke tests, Stage 3 sequential ablations (SpecAugment-only, dropout-only, conditionally weight-decay-only; broader-LoRA; full-decoder-FT if feasible), Stage 4 fair comparison, Stage 5 analysis of already-completed [[EXP-007]] per-speaker breakdown.

Random Seed: 1234 (matching existing convention); single seed except the final chosen headline configuration, which gets one confirmatory re-seed run (Stage 7).

Hardware: 4× RTX 2080 Ti, 11GB each — verified this session (`nvidia-smi`), unchanged from prior experiments.

GPU Allocation: 4 GPUs via FSDP (`tensor_parallel_size=1, data_parallel_size=4`), per [[DEC-004]] — DDP OOMs this model size.

Command: Not yet finalized — pending Stage 0 spike results (see report Section 16 for the adaptive decision tree).

Estimated Cost: ~86–108 GPU-hours for the full P0 program (~5–6 wall-clock days at 4-GPU utilization), per the report's Section 17 compute budget.

Actual Runtime: N/A — not executed.

Results: N/A — not executed.

Metrics: N/A — not executed.

Conclusion: N/A — not executed. No training has been launched for this experiment; this record exists per the Experiment Gate rule (proposed, not implying execution) to make the proposed program visible to future sessions before any GPU-hours are spent.

Next Action: Await explicit user approval before Stage 0 spikes (which are free/code-only) or any subsequent GPU-hour-consuming stage. See `research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` for full design, risk register, and adaptive decision tree.

Related Records: [[DEC-005]], [[ISS-007]], [[EXP-007]], [[ISS-005]], [[ISS-006]], [[AUD-004]]

---

## EXP-008 — UWB-ATCC W2V2 hyperparameter ablations (dropout, mask_time_prob)

Date: recovered 2026-09-06 (proposed; not yet executed)
Status: PROPOSED

Objective: Study the effect of dropout and mask_time_prob on UWB-ATCC W2V2 fine-tuning WER.

Evidence of proposal only: SUMMARY.md "Pending / Next Steps" — "UWB-ATCC W2V2 ablations: Effect of dropout, mask_time_prob on UWB-ATCC W2V2." No configuration, run, or result exists in the repo.

Related Records: [[EXP-001]]

---

## EXP-010 — Execution-ready adaptive research program (IEEE SLT review response, full plan)

Date: 2026-09-07 (proposed; not yet executed)
Status: PROPOSED

Objective: Convert [[DEC-005]]'s chosen research question (Canary-Qwen decoder-adaptation-scope study) into a staged, gated, GPU-hour-minimizing execution plan. Full design in `research_report/FINAL_RESEARCH_PROGRAM.md`.

Hypothesis: Broadening Canary-Qwen's decoder adaptation scope beyond LoRA q/v-only will measurably reduce UWB-ATCC WER; the magnitude determines whether the redesigned paper's headline claim is "adaptation scope explains most of the W2V2 gap" (best case), "adaptation scope explains part of it" (neutral case), or "adaptation scope alone does not explain it" (negative case) — see report Section 14.

Dataset: UWB-ATCC (train=10.54h/11,543 utt, test=2.63h/2,886 utt — verified this session) for the core adaptation-scope/regularization program; ATCOSIM gender-stratified splits (verified leakage-free, [[EXP-007]]) for the speaker-independent capstone, pending [[ISS-008]]'s lhotse-cuts prerequisite.

Model: nvidia/canary-qwen-2.5b (SALM), Qwen3-1.7B decoder, FastConformer encoder (frozen throughout this program).

Configuration: Staged — Stage 0 (audit, done) → Stage 1 (Spike D: 1-step VRAM feasibility smoke test, ~0.02 GPU-h, requires explicit approval to launch) → Stage 2 (fresh v1-equivalent control checkpoint, ~21 GPU-h) → Stage 3 (adaptation-scope Branch 2 broader-LoRA always, Branch 3 full-decoder-FT gated on Stage 1; regularization Runs 1/2 always, Run 3 gated on Run 1's result) → Stage 4 (decoding-fairness N-best+KenLM comparison, inference-only; speaker-independent capstone on winning config) → Stage 5 (optional confirmatory re-seed) → Stage 6 (manuscript reconstruction, no GPU cost).

Random Seed: 1234 (matching existing convention); single seed except the final headline configuration (Stage 5 re-seed).

Hardware: 4× RTX 2080 Ti, 11GB each — verified this and prior session. FSDP (`tensor_parallel_size=1, data_parallel_size=4`) per [[DEC-004]], unchanged from prior Canary-Qwen runs — not reduced below 4 GPUs, since doing so would invalidate comparison with existing baseline numbers measured under this exact topology.

GPU Allocation: 4 GPUs for every training experiment; 1 (inference-only, no training) for the decoding-fairness comparison.

Command: Not finalized — pending Stage 1's Spike D result (see report Section 4's adaptive decision tree for the full branching logic).

Estimated Cost: Minimum ~63 GPU-hours (~3.4 wall-clock days), expected ~94.5 GPU-hours (~5.1 days), maximum ~126 GPU-hours (~6.75 days) — full breakdown in report Section 17.

Actual Runtime: N/A — not executed.

Results: N/A — not executed.

Metrics: N/A — not executed.

Conclusion: N/A — not executed. This record exists per the Experiment Gate rule to make the proposed, gated program visible to future sessions before any GPU-hours are spent. Supersedes [[EXP-009]] as the execution-ready version of the same underlying research question — [[EXP-009]] is preserved unchanged as the earlier, less-structured proposal.

Next Action: Await explicit user approval for Stage 1's Spike D (the first step requiring GPU time, ~0.02 GPU-hours, <30 min) — per the report's executive summary, this is recommended as the first action "tomorrow morning."

Related Records: [[DEC-005]], [[DEC-006]], [[ISS-007]], [[ISS-008]], [[VAL-008]], [[EXP-007]], [[EXP-009]], [[AUD-004]]

---

## EXP-011 — Re-run the "lower-LR" (1e-4) Canary-Qwen UWB-ATCC ablation (low priority)

Date: 2026-09-08 (proposed; not yet executed)
Status: PROPOSED, LOW PRIORITY

Objective: Re-establish a verifiable checkpoint for the manuscript-cited "lower-LR" ablation (lr=1e-4, otherwise identical to v1's LoRA config), whose original checkpoint is confirmed lost (`run_0`, logs only, per [[VAL-012]]).

Hypothesis: A fresh run under `salm_uwb_atcc_lr1e4.yaml` (currently local-only, not yet committed to GitHub) will reproduce the cited 32.58% WER, consistent with how v1's fresh reproduction matched its historical citation almost exactly ([[VAL-010]]).

Dataset: UWB-ATCC (same as v1/v2/v3 — train=10.54h/11,543 utt, test=2.63h/2,886 utt).

Model: nvidia/canary-qwen-2.5b (SALM), LoRA q_proj+v_proj r=128, lr=1e-4 (vs. v1's 5e-4), no SpecAugment, weight_decay=1e-3, 10,000 steps.

Configuration: `salm_uwb_atcc_lr1e4.yaml` (needs to be committed to `models/canary-qwen/scripts/` as part of this experiment, and renamed to fit the v1/v2/v3-adjacent convention — e.g. `salm_uwb_atcc_lr1e4.yaml` is fine as a distinctly-named variant, not part of the core v1/v2/v3 trio).

Hardware: 4× RTX 2080 Ti, FSDP (`tensor_parallel_size=1, data_parallel_size=4`), matching every other Canary-Qwen UWB-ATCC run this session.

Estimated Cost: ~21 GPU-hours (~5.25h wall-clock on 4 GPUs), based on the verified actual rate from [[VAL-010]]'s fresh v1 reproduction (not the previously-documented-but-wrong ~5.3h figure).

Priority: **Low** — per user instruction, this is queued behind the higher-priority items in [[EXP-010]]'s execution plan (Spike D, the adaptation-scope study, the regularization ablation), not scheduled for immediate execution.

Actual Runtime: N/A — not executed.

Results: N/A — not executed.

Next Action: Await explicit approval, scheduled after [[EXP-010]]'s higher-priority items.

Related Records: [[VAL-012]], [[EXP-010]], [[DEC-007]]

## EXP-012 — S4-FAIR: decoding-fairness comparison (N-best + in-domain KenLM on Canary-Qwen)

Date: 2026-09-08
Status: COMPLETE

Objective: Resolve reviewer complaint that the manuscript compared W2V2+KenLM against Canary-Qwen native decoding (apples-to-oranges) by decoding Canary-Qwen with the same in-domain KenLM, via N-best generation + rescoring.

Hypothesis: If the fairness asymmetry is a major driver of the W2V2-Canary WER gap, giving Canary-Qwen the same KenLM should close most of the gap. If adaptation scope is the real driver, KenLM should barely move Canary-Qwen's WER.

Dataset: UWB-ATCC full test set (2,886 utterances).

Model: Canary-Qwen v1 checkpoint ([[VAL-010]]), 5-beam/5-best generation.

Configuration: `generate_nbest.py` (5-beam, `output_scores=True`), `rescore_kenlm.py` (KenLM fusion, alpha grid, headline alpha=0.5 matching W2V2's own pyctcdecode default).

Hardware: 1× RTX 2080 Ti (inference-only).

Command: `python generate_nbest.py --checkpoint step=10000-last.ckpt --test-manifest test_manifest.json --num-beams 5` then `python rescore_kenlm.py --nbest nbest_v1_full.json --kenlm uwb_atcc_4g.binary`.

Estimated Cost: ~0 GPU-hours (inference-only, ~30 minutes wall-clock).

Actual Runtime: ~30 minutes (generation) + seconds (rescoring).

Results: See [[VAL-013]] for full table. Headline: Canary-Qwen v1 native 23.32% -> 5-beam 22.28% -> +KenLM (alpha=0.5) 21.79%. W2V2: 14.54% native / 12.69% +KenLM.

Conclusion: The fairness fix closes only ~1.5 WER points, most of it from beam search rather than the LM itself. The W2V2-Canary gap is not substantially explained by the decoding-fairness asymmetry, supporting the manuscript's adaptation-scope framing.

Next Action: Report this comparison in the manuscript's decoding-fairness section; no further action needed unless a reviewer requests dev-set-tuned alpha.

Related Records: [[VAL-013]], [[VAL-008]], [[VAL-010]], [[DEC-002]]

## EXP-013 — S4-FAIR: decoding-fairness comparison (N-best + in-domain KenLM on Canary-Qwen v3)

Date: 2026-09-08
Status: COMPLETE

Objective: Repeat EXP-012's decoding-fairness comparison on the v3 (regularized) checkpoint, the manuscript's headline result.

Hypothesis: Same as EXP-012 — external KenLM should meaningfully help if the fairness asymmetry drives the WER gap.

Dataset: UWB-ATCC full test set (2,886 utterances).

Model: Canary-Qwen v3 checkpoint ([[VAL-012]]), 5-beam/5-best generation.

Configuration: Same `generate_nbest.py`/`rescore_kenlm.py` as EXP-012, unmodified.

Hardware: 1x RTX 2080 Ti (inference-only).

Estimated Cost: ~0 GPU-hours.

Actual Runtime: ~30 minutes generation + seconds rescoring.

Results: See [[VAL-014]]. v3 native greedy 20.70% -> 5-beam 19.42% -> +KenLM (alpha=0.5) 20.14% (WORSE than beam alone). Best point in the sweep (alpha=0.1, 19.44%) is a tie with no-LM beam search, not an improvement.

Conclusion: Unlike v1 (EXP-012), KenLM provides no benefit to v3 and is actively harmful at the pre-registered headline alpha. All of v3's improvement over greedy comes from beam search. Reported honestly rather than substituting a post-hoc favorable alpha.

Next Action: None required — this negative result stands as reported. Worth noting in the manuscript that decoding-fairness correction does not help the regularized configuration.

Related Records: [[VAL-014]], [[EXP-012]], [[VAL-012]]

## EXP-014 — Full multi-version WER-vs-true-epoch learning curve (v1/v2/v3 LoRA vs all S3-B3 full-decoder attempts)

Date: 2026-09-14
Status: COMPLETE

Objective: Build a single, fair (true-data-epoch-normalized, not raw-step-normalized) learning curve across every trained version in this project, per user request ("get the curve, but not just for S3-B3 but all versions trained... we did this in the paper").

Context: v1/v2/v3's existing per-step WER curves (`models/canary-qwen/docs/learning_curve_v{1,2,3}.json`, produced by `eval_learning_curve.py`, dated 2026-04-03/04/22-23, i.e. run sequentially on different days -- no cache-collision risk) were reused as-is after verifying their decoding path is deterministic (`--base released` composes onto canary-qwen-2.5b's own LoRA-shaped GenerationConfig, which per ISS-013's follow-up keeps `do_sample=False` by default for this specific path -- confirmed all curve points show `samples:500`, no silent generation failures). S3-B3's curve required 4 new evals to fill gaps: step=375/625 (500-sample, same protocol as VAL-019/022/023) and step=500/2000 (FULL 2,886-sample test set, to get a statistically tight number for the two most decision-relevant checkpoints).

**Critical methodological point this analysis surfaced**: v1/v2/v3 use `train_cuts.jsonl.gz` (11,543 cuts, effective batch 32) -> 360.7 optimizer-steps/true-epoch. S3-B3 uses `train_cuts_v2.jsonl.gz` (10,619 cuts, effective batch 32) -> 331.8 optimizer-steps/true-epoch. Every WER comparison made earlier this session (VAL-019, VAL-022, VAL-023, and both prior root-cause-analysis agents) compared these arms at matched RAW STEP COUNTS or treated "step=625" as directly comparable across families, without normalizing for this ~8.7% rate difference AND without accounting for the far larger confound that v1/v2/v3 trained to step=10000 (27.72 true epochs) while S3-B3 was capped at step=2000 (6.03 true epochs) -- an unequal-training-exposure confound larger than any of the previously-discussed ones (init, optimizer, data version, eval-set).

Procedure: Converted every existing and newly-measured (step, WER) pair to true_epochs = step / steps_per_epoch for its family, tabulated below.

Actual Result (WER%, true epochs in parentheses):

| true epochs (approx) | v1 (LoRA) | v3 (LoRA+reg) | S3-B3 broken-init | S3-B3 corrected/uncalib | S3-B3 corrected/recalibrated |
|---|---|---|---|---|---|
| ~1.1-1.4 | 39.14 (1.39) | 39.51 (1.39) | -- | -- | 28.26 (1.13) |
| ~1.5 | -- | -- | -- | -- | **26.03 (1.51, FULL 2886-set)** |
| ~1.9-2.5 | -- | -- | 39.92 (2.54) | 27.48 (1.88) | 26.26 (1.88) |
| ~5.5-6.0 | 30.87 (5.54) | 27.28 (5.54, interpolated point not in original table -- see raw learning_curve_v3.json for step=2000 at 27.28%, true_epochs=5.54, closest available) | -- | -- | **24.12 (6.03, FULL 2886-set, FINAL checkpoint)** |
| ~13.9 | 24.77 (13.86) | 23.00 (13.86) | -- | -- | -- |
| ~27.7 (canonical, full test set) | **23.32** | **20.70** | -- | -- | -- |

Statistical check on the two full-set S3-B3 numbers (step=500: 26.03%, n=2886; step=2000: 24.12%, n=2886): unpaired normal-approximation z-test, diff=1.90pp, z=1.67, two-tailed p≈0.095 -- borderline, not conventionally significant, but the SAME direction and near-identical magnitude replicated from the original n=500 estimates (26.08%/24.06%), i.e. not an artifact of small-sample noise. A paired test (same utterances, both checkpoints) would likely show tighter significance but requires per-utterance data not currently saved by `eval_finetuned.py`.

Result: **At matched true training exposure (~5.5-6 epochs), the corrected-init full-decoder S3-B3 arm (24.12%) already beats v1 (30.87%) and is within ~1.1pp of v3 (23.00%) despite v3 having ~2.3x more epochs of training.** This directly contradicts the framing (used throughout this session prior to this analysis, including by two independent root-cause-diagnosis agents) that full-decoder "underperforms LoRA" -- that framing implicitly compared S3-B3's early-stopped ~6-epoch number against v1/v3's fully-converged ~27.7-epoch canonical numbers, an unequal-exposure comparison. It remains unknown whether continuing S3-B3 to a matched ~27.7-true-epoch budget (~9,200 steps at S3-B3's rate) would continue improving, plateau, or (per the already-documented capacity/regularization concerns) resume degrading -- this is a genuinely open question, not yet answered by any run in this project.

Conclusion: The planned next matched-protocol comparison (LoRA arm + full-decoder arm, per the converged agent plan) should NOT reuse the originally-proposed max_steps=2000 (6.03 true epochs) as if it were a complete picture -- it under-samples the training exposure both arms received historically for v1/v3's canonical numbers by more than 4x. Whether to budget the next comparison at the shorter (cheap, ~13-19h) 2000-step scale or the longer (~4.6x cost) ~9,200-step scale to match historical exposure is a real cost/completeness tradeoff requiring an explicit decision, not a default.

Next Action: Present this tradeoff to the user before launching the matched-protocol arms. Do not assume the 2000-step budget from the earlier agent-authored plan is still the right choice now that this confound is known.

Related Records: [[VAL-019]], [[VAL-022]], [[VAL-023]], [[DEC-010]]

## EXP-015 — Matched-protocol production runs: all three arms COMPLETE and evaluated (full-decoder 9200 steps, LoRA-truncated 3700 steps, LoRA-full 9200 steps)

Date: 2026-09-17 to 2026-09-22
Status: ALL THREE ARMS COMPLETE, EVALUATED (dev + one-time test touch each)

Objective: Execute DEC-011's decided matched-protocol comparison (full-decoder vs LoRA, identical init/optimizer/data/schedule/regularization, only adaptation scope differing) using the LR values selected in [[VAL-025]] (full-decoder lr=2e-5, LoRA lr=5e-4).

Configuration: `salm_uwb_atcc_matched_full_decoder.yaml` (max_steps=9200, full 27.72 true epochs) and `salm_uwb_atcc_matched_lora.yaml` (max_steps=3700, truncated to ~11.15 true epochs per [[DEC-012]] for the ICASSP 2027 deadline). Both: 4 GPUs, FSDP2 (`ModelParallelStrategy`, `data_parallel_size=4`), `train_cuts_v2.jsonl.gz`, spec_augment enabled, weight_decay=1e-2, `load_released_pretrained=True`.

Full-decoder arm training: Launched 2026-09-17 00:22. Interrupted once at step ~5669 by an unrelated operational error (see [[ISS-016]]) with the last-good checkpoint (step=5500) intact; resumed and completed cleanly to step 9200 on 2026-09-19 10:03 after working around three resume-path bugs (ISS-016). Checkpoints saved at steps 2300/4600/6900/9200 (`save_top_k=-1`, all preserved). val_loss trajectory: step 750 (2.26 true epochs) = 0.562 (minimum), drifting up and plateauing in the 0.56-0.65 band through step 9200 (final = 0.623, best-observed-post-resume = 0.621 at step 6900).

LoRA-truncated arm training: Launched 2026-09-19 10:04 immediately after full-decoder's GPUs freed. Completed cleanly to step 3700 on 2026-09-20 02:31, ~16s/step (29.5% faster per step than full-decoder's ~22.7s/step, measured directly from checkpoint timestamps). Checkpoints at steps 925/1850/2775/3700. val_loss trajectory: 925=0.621, 1850=0.595 (minimum), 2775=0.603, 3700=0.610 -- a fully re-annealed, converged short schedule (cosine LR reached its floor at step 3700), not a mid-trajectory snapshot of a longer run.

**Dev-set WER (full 915-sample set, all 8 checkpoints, `eval_finetuned.py --base composed`, no `--max-samples` cap):**

| Arm | Step | True epochs | val_loss | Dev WER |
|---|---|---|---|---|
| Full-decoder | 2300 | ~6.93 | -- | 18.80% |
| Full-decoder | 4600 | ~13.86 | -- | 18.32% |
| Full-decoder | 6900 | ~20.79 | 0.621 (best) | 18.87% |
| Full-decoder | **9200** | **27.72** | 0.623 | **18.19% (arm winner)** |
| LoRA-truncated | 925 | ~2.79 | 0.621 | 25.20% |
| LoRA-truncated | 1850 | ~5.58 | 0.595 (best) | 20.59% |
| LoRA-truncated | 2775 | ~8.36 | 0.603 | 20.00% |
| LoRA-truncated | **3700** | **~11.15** | 0.610 | **19.83% (arm winner)** |

Both arms' final (highest-exposure) checkpoint won on dev WER, despite val_loss plateauing/drifting earlier in both cases -- consistent with the val_loss/WER decoupling established in [[VAL-023]]/[[VAL-024]]. Full-decoder's checkpoint-to-checkpoint WER is NOT monotonic (18.80/18.32/18.87/18.19), a ~0.68pp spread that is a meaningful fraction of the eventual arm-to-arm gap -- disclose this, do not present the comparison as noise-free.

**Test-set WER (full 2,886-sample set, ONE touch per arm's dev-selected winner, per [[VAL-017]]/[[DEC-011]] leakage-avoidance):**

| Arm | Checkpoint | True epochs | Test WER |
|---|---|---|---|
| Full-decoder | step=9200 | 27.72 | **18.73%** |
| LoRA-truncated | step=3700 | ~11.15 | **20.62%** |

Full-decoder beats LoRA-truncated by 1.89pp on the held-out test set under fully matched conditions -- the clean, confound-controlled result [[DEC-011]] was designed to produce. Per explicit user decision (2026-09-20), the truncated-LoRA test touch was NOT discarded even though a full-length LoRA run was subsequently also launched (see below) -- both are being kept and reported as separate, distinctly-configured experimental arms, each with its own single test touch, rather than treating the truncated run as superseded.

LoRA-full arm training (reversing DEC-012's truncation, see [[DEC-013]]): Launched 2026-09-20 14:14 (`experiments_matched_lora_full9200/`, separate log dir from the truncated run, no collision) with two safeguards added after two-agent pre-launch review: `exp_manager.checkpoint_callback_params.every_n_train_steps=1150` (aligns checkpoints with full-decoder's own 2300/4600/6900/9200) and `exp_manager.create_early_stopping_callback=false` (proactively avoids the ISS-015/ISS-016 EarlyStopping cross-device crash). Completed cleanly to step 9200 on 2026-09-22 07:11, ~16.3s/step throughout, zero errors, DEC-013's bailout point (step 6900 by 2026-09-22 00:00 EDT) reached comfortably early and never needed. Checkpoints at steps 1150/2300/3450/4600/5750/6900/8050/9200. val_loss trajectory: bottoms at step 1150 (0.595), plateaus/drifts in the 0.58-0.65 band thereafter (final=0.639) -- same pattern as both other arms.

**LoRA-full dev-set WER (full 915-sample set, all 8 checkpoints):**

| Step | True epochs | Dev WER |
|---|---|---|
| 1150 | ~3.47 | 22.07% |
| 2300 | ~6.93 | 22.10% |
| 3450 | ~10.40 | 22.29% |
| 4600 | ~13.86 | 20.42% |
| 5750 | ~17.33 | 18.96% |
| 6900 | ~20.79 | 19.48% |
| 8050 | ~24.26 | 17.90% |
| **9200** | **27.72** | **17.83% (arm winner)** |

Like both other arms, the final (highest-exposure) checkpoint wins on dev WER despite val_loss having bottomed much earlier (step 1150). Notably, LoRA-full's dev-set winner (17.83%) is BETTER than full-decoder's own dev-set winner (18.19%) -- an early signal (confirmed below on test) that at fully matched 27.72-epoch exposure, the arm-to-arm gap seen in the truncated comparison narrows and may not hold the same direction dev-side.

**FINAL matched-protocol comparison, all three arms, test set (2,886 samples, one touch per arm's dev-selected winner):**

| Arm | Steps | True epochs | Test WER |
|---|---|---|---|
| Full-decoder | 9200 | 27.72 | **18.73%** |
| LoRA-truncated | 3700 | ~11.15 | 20.62% |
| **LoRA-full** | **9200** | **27.72** | **19.70%** |

At fully matched exposure (both arms 27.72 true epochs, identical init/optimizer/data/regularization/schedule length, only adaptation scope differing), full-decoder still beats LoRA, but the gap narrows from 1.89pp (vs. the truncated LoRA run) to **0.97pp** (vs. LoRA-full) -- the truncated comparison somewhat overstated the full-decoder advantage, since it compared full-decoder at its full exposure against LoRA at less than half that exposure. This 0.97pp gap at matched exposure is the cleanest, most defensible headline number for the manuscript. All three arms are reported (not just full-decoder vs. LoRA-full) since the truncated run is itself informative about the exposure-sample-efficiency question this project has tracked since [[EXP-014]].

Next Action: Write up all three arms in the manuscript with their distinct true-epoch exposures made explicit; report the full 8-checkpoint dev tables for both LoRA arms and full-decoder as supplementary evidence of the non-monotonic/plateau-then-recover WER pattern; disclose the checkpoint-selection-instability finding (full-decoder's 0.68pp non-monotonic spread; LoRA's own checkpoints also non-monotonic, e.g. step 6900's 19.48% is worse than step 8050's 17.90%) as a limitation alongside the headline comparison.

Related Records: [[VAL-025]], [[DEC-011]], [[DEC-012]], [[DEC-013]], [[ISS-016]], [[EXP-014]], [[AUD-006]]
