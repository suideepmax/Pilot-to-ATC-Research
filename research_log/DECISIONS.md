# Engineering Decisions

## DEC-001 — Discard ATCOSIM gender-based WER results; require a re-split for speaker independence

Date: 2026-09-06 (recovered; original decision made during Phase 4 of ATCOSIM work)
Status: DECIDED — corrected experiment still PENDING (not yet executed in this repo)

Question: Can the ATCOSIM gender subsets (test_female/test_male) be used to report a valid speaker-independent WER for the full-corpus-trained model?

Context: The full ATCOSIM W2V2-large model was trained on a random 80/20 split spanning all 10 speakers. Gender subsets are speaker-filtered views of that same corpus (see [[ISS-001]], [[AUD-001]]).

Evidence: ~80% of each gender-subset speaker's utterances were already in the training data; discarded numbers were 0.86%/0.01% (custom script) and 86%/1.23% (eval_model.py, also affected by [[ISS-002]]).

Options Considered:
1. Report the gender-subset numbers as "unseen speaker" WER anyway.
2. Discard the numbers and require a proper speaker-disjoint re-split before making any speaker-independence claim.

Decision: Option 2 — discard the leaked numbers, keep them on record marked invalid, and require training a new model on `train_male`/`train_female` splits evaluated only against the disjoint held-out speakers before reporting a speaker-independent ATCOSIM WER.

Reason: Reporting the leaked numbers would materially overstate model generalization (0.01%–0.86% WER is not credible as a "held-out speaker" result once ~80% of that speaker's data was already seen during training).

Consequences: The repo's ATCOSIM "1.67%/1.28%" headline numbers remain valid only as "same-corpus, random-split" results, not as evidence of speaker-generalization; this caveat is explicit in SUMMARY.md and PROGRESS_ATCOSIM.md.

Rejected Alternatives: Reporting the leaked numbers with a caveat footnote was considered insufficient given how easily such numbers get copied without caveats.

Related Records: [[ISS-001]], [[AUD-001]], [[EXP-007]] (proposed, not yet run)

---

## DEC-002 — Canonicalize "greedy" vs "beam search" terminology and pick single canonical WER numbers

Date: 2026-06-11 (commit 34a5ca1), recovered 2026-09-06
Status: DECIDED and APPLIED

Question: Docs across the repo referred to the UWB-ATCC no-LM decoding as "beam search," and reported an averaged WER (14.57%/12.76%) across two independent standalone-eval runs of the same checkpoint. Which terminology and which number should be canonical?

Context: `eval_model.py` performs greedy (argmax) CTC decoding when no LM is given, and beam search only when a KenLM is attached alongside beam search decoding. The "beam search, no LM" label in earlier docs was a misnomer. Separately, two standalone eval runs of the same final checkpoint gave 14.54%/12.69% and 14.60%/12.82% — a small run-to-run variation, and docs had been averaging them.

Evidence: `git show 34a5ca1` diff across SUMMARY.md, PROGRESS_UWB_ATCC.md, PROGRESS_ATCOSIM.md.

Options Considered:
1. Keep reporting the 2-run average (14.57%/12.76%) labeled "beam search."
2. Correct the label to "greedy" and report the single canonical run stored in `finetuned_results_v2.json` (14.54%/12.69%), keeping the second run's numbers as a documented note on variance.

Decision: Option 2.

Reason: The label must match what the code actually does (greedy, not beam search) to avoid misleading comparisons against the paper's own beam-search numbers. The canonical single-run number was preferred over an ad hoc average because `finetuned_results_v2.json` is the artifact of record, and averaging two runs without a defined protocol invites confusion.

Consequences: All current docs (README.md, SUMMARY.md, REPLICATION_GUIDE.md, PROGRESS_UWB_ATCC.md) report UWB-ATCC W2V2 = 14.54% (no LM, greedy) / 12.69% (CTC+KenLM) as canonical, with the 14.60%/12.82% run noted separately as evidence of run-to-run variation.

Rejected Alternatives: Averaging retained as an option was rejected for lack of a defined multi-run averaging protocol (only 2 runs existed, not a statistically established set).

Related Records: [[AUD-003]]

---

## DEC-003 — Use torchrun/DDP instead of the paper's DataParallel launcher for W2V2-large training

Date: recovered 2026-09-06 (original decision made during Phase 4 of UWB-ATCC work)
Status: DECIDED and APPLIED

Question: The paper's `src/run_asr_fine_tuning.sh` launches training via `python3`, which triggers PyTorch DataParallel (DP). Should this be kept for fidelity to the paper, or changed?

Context: DP copies the full 317M-parameter model to every GPU and gathers gradients on GPU 0, which OOMs on the available 11GB RTX 2080 Ti GPUs.

Evidence: models/w2v2/docs/PROGRESS_UWB_ATCC.md Phase 4 "Script Modifications Required"; REPLICATION_GUIDE.md Part 3 known-issues table ("DDP OOM with 2.5B model" is the Canary analog; the W2V2 case is the DP OOM entry).

Options Considered:
1. Keep DataParallel, reduce batch size further to try to fit on GPU 0.
2. Switch to `torchrun` for DistributedDataParallel (DDP), where each GPU handles only its own gradients via ring all-reduce.

Decision: Option 2 — switched to `torchrun --nproc_per_node=4` DDP.

Reason: DDP does not require gathering the full model/gradients onto a single GPU, avoiding the OOM without further crippling batch size. This is a hardware-driven deviation from the paper's exact training method, explicitly documented as such (paper: 1 GPU, DataParallel, effective batch 24; here: 4 GPUs, DDP, effective batch 64).

Consequences: Effective batch size differs from the paper (64 vs paper's 24 for UWB-ATCC, 64 vs paper's 96 for ATCOSIM); learning rate was also raised to 5e-4 (paper used 1e-4). Results still beat the paper's numbers on both corpora, but the run is explicitly a "not identical to the paper" replication, not a strict reproduction (see PROGRESS_ATCOSIM.md: "Training was NOT identical to the paper.").

Rejected Alternatives: Further batch-size reduction under DP was not pursued once the OOM's DP-specific root cause was identified.

Related Records: [[ENV-002]], [[EXP-001]], [[EXP-002]]

---

## DEC-004 — Use FSDP (ModelParallelStrategy) instead of DDP for Canary-Qwen-2.5B training

Date: recovered 2026-09-06
Status: DECIDED and APPLIED

Question: The 2.87B-parameter Canary-Qwen model needs a multi-GPU strategy on 4x 11GB GPUs. Should DDP (as used for W2V2) be reused?

Context: DDP requires each GPU to hold a full copy of the model plus optimizer state; a 2.87B-parameter model (with the LLM decoder) does not fit in 11GB per GPU under DDP.

Evidence: REPLICATION_GUIDE.md Part 3 ("DDP OOM with 2.5B model → Use ModelParallelStrategy (FSDP)"); models/canary-qwen/docs/PROGRESS.md "Known Issues & Fixes."

Options Considered:
1. DDP (as used for W2V2).
2. FSDP via NeMo/Lightning's `ModelParallelStrategy` (tensor_parallel=1, data_parallel=4), sharding model/optimizer state across GPUs.

Decision: Option 2 — FSDP.

Reason: FSDP shards parameters and optimizer state across GPUs instead of replicating them, fitting the larger model in available VRAM.

Consequences: `ModelParallelStrategy` rejects `16-mixed` precision, forcing `16-true`; this in turn required AdamW `eps=1e-4` to avoid NaN (see [[ISS-003]]). NeMo's FSDP path also does not log metrics to stdout normally, requiring val_loss extraction from checkpoint messages.

Rejected Alternatives: DDP was ruled out directly by the OOM evidence, not by trial.

Related Records: [[ENV-003]], [[ISS-003]]

---

## DEC-005 — Redesign the paper's research question around a controlled Canary-Qwen decoder-adaptation-scope study, dropping the architecture-vs-architecture and "unseen domain" framing

Date: 2026-09-07
Status: PROPOSED (planning only — no training launched, no manuscript edited)

Question: The submitted manuscript (`Final_Draft.md`) was rejected by all 4 IEEE SLT 2026 reviewers (2/5 each), with 3 of 4 independently flagging that the W2V2-vs-Canary-Qwen comparison cannot isolate architecture/scale/pretraining/adaptation-scope/decoding effects, and that "unseen domain" mischaracterizes an in-domain 80/20 split. What should the redesigned research question be?

Context: Full analysis in `research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md`. Four candidate research questions were evaluated (data-efficiency crossover, decoder adaptation-scope study, speaker/cross-corpus generalization, compute-efficiency tradeoff) against available hardware (4×11GB GPUs), lost checkpoints (full retraining required regardless), and the actual verified feasibility of the installed Canary-Qwen/NeMo pipeline.

Evidence: Manuscript text (reconciled directly against reviews, repo, and configs — see report Sections 2–4); verified config mechanism for decoder LoRA (`salm_uwb_atcc.yaml`: `freeze_params` + `lora.target_modules: [q_proj, v_proj]`, standard PEFT injection over frozen base weights); [[ISS-007]] (encoder-unfrozen provenance problem); [[EXP-007]] (speaker-independent ATCOSIM results, already fulfilling the manuscript's own stated future work).

Options Considered: (A) data-efficiency crossover across data scales; (B) controlled decoder-adaptation-scope study on Canary-Qwen (LoRA q/v-only → broader LoRA → full decoder fine-tune, gated on a VRAM feasibility check); (C) speaker-independence + cross-corpus generalization framing; (D) compute-efficiency-vs-accuracy tradeoff framing.

Decision: Adopt (B) as the primary research question, with (C)'s speaker-independence half (already substantially complete via [[EXP-007]]) and (D)'s efficiency framing as secondary supporting analyses. Cross-corpus generalization (the other half of C) is deprioritized to OPTIONAL/exploratory given UWB-ATCC and ATCOSIM differ on sample rate, recording conditions, vocabulary, and speaker population simultaneously — a naive cross-corpus number would conflate too many confounds to be interpretable as a headline claim.

Reason: (B) is the only candidate that is (1) fully executable with existing code and modest compute (~86–108 GPU-hours for the P0 program), (2) directly tests a claim the current manuscript makes without evidence (the "frozen decoder is the bottleneck" interpretation in Sec. VI.B/F), (3) is honestly scoped — it does not attempt to resolve the cross-architecture question the reviewers correctly say can't be resolved with these two models, and (4) is explicitly identified by Reviewer 4Pd7 as a gap ("Decoder-LoRA is central to the paper's interpretation and is currently future work").

Consequences: The paper's headline claim changes from "W2V2 beats Canary-Qwen" to "adaptation scope significantly affects Canary-Qwen's ATC performance, and [does/does not] close the gap to a fully fine-tuned CTC baseline" — the exact wording depends on the outcome of the full-decoder-fine-tune experiment (S3-5 in the report's roadmap), which is not yet known. The Acknowledgements/AI-disclosure section will need updating if EXP-007 results are used, since Claude directly executed those runs (unlike the original paper's disclosed AI-use boundary).

Rejected Alternatives: (A) data-efficiency crossover — high value but expensive if run on both corpora at multiple scales; kept as a SHOULD-TEST single-corpus, few-scale version rather than the primary axis. (D) alone — doesn't resolve the core reviewer complaint, only reframes it; folded in as secondary analysis instead.

Related Records: [[ISS-007]], [[EXP-007]], [[ISS-005]], [[ISS-006]], [[DEC-003]], [[DEC-004]], [[AUD-004]]

---

## DEC-006 — Drop data-scale and cross-corpus studies from the P0/P1 execution-ready program

Date: 2026-09-07
Status: DECIDED (planning only — no training launched)

Question: The reviewers explicitly asked "how much in-domain data does Canary-Qwen need to surpass W2V2?" (data-scale) and criticized "unseen domain" as unsupported without cross-corpus evaluation. Should either be included in the P0 execution-ready program (`research_report/FINAL_RESEARCH_PROGRAM.md`)?

Context: Full reasoning in `FINAL_RESEARCH_PROGRAM.md` Sections 8 and 10.

Evidence: Data-scale — even a minimal 2-3-scale design would roughly double the compute budget of the chosen adaptation-scope study (Section 2/6) without directly serving that research question. Cross-corpus — UWB-ATCC (8kHz native, real noisy comms, broad vocabulary) and ATCOSIM (32kHz→16kHz, clean scripted, narrow vocabulary) differ on sample rate (W2V2 currently consumes UWB-ATCC at native 8kHz per the manuscript, Sec. III.B — a cross-corpus W2V2 run would need a new resampling step not present in any current script), unverified transcript-normalization compatibility, and vocabulary/channel confounds stacked together, making any resulting cross-corpus WER difficult to attribute to "domain shift" specifically.

Options Considered: (1) include both studies at reduced scope; (2) include data-scale only; (3) include cross-corpus only (one direction); (4) drop both, state as future work.

Decision: Drop both (option 4) from the P0/P1 program.

Reason: The chosen research question (decoder adaptation scope, [[DEC-005]]) is deliberately narrow and focused — Reviewer X1FA specifically criticized the original manuscript for lacking focus ("the main research question and take-away are not sufficiently focused"). Adding either study would split the paper's attention across two research questions rather than answering one well. Both remain legitimate future-work items and should be stated as such in the redesigned manuscript's Limitations/Future Work section, not silently dropped without acknowledgment.

Consequences: The redesigned paper's generalization claim is limited to same-corpus, speaker-disjoint evaluation (via [[EXP-007]] and its planned Canary-Qwen capstone) — it does not claim cross-corpus or data-efficiency evidence. This is a narrower but more defensible scope than the original manuscript attempted.

Rejected Alternatives: Reduced-scope versions of both (option 1) were considered but rejected as still diluting focus for partial evidence in either direction — better to fully commit compute to one well-powered study (adaptation scope + regularization) than to split it three ways.

Related Records: [[DEC-005]], [[EXP-010]], [[EXP-007]]

---

## DEC-007 — Drop the "research-optimized" (3e-5) ablation from the active manuscript and research record

Date: 2026-09-08
Status: DECIDED (user instruction)

Question: The manuscript cites a "research-optimized" (lr=3e-5, 2,500 steps) Canary-Qwen ablation at 60.46% WER. Investigation found this citation is ambiguous between two different local configs (`salm_uwb_atcc_optimized.yaml`, r=64/4-projection LoRA, which crashed with NaN and was never evaluable per `PROGRESS.md`; and `salm_uwb_atcc_optimized_v2.yaml`, r=128/2-projection, which produced the citation) — neither has a surviving checkpoint. Should this ambiguity be resolved by retraining, or should the ablation be dropped?

Context: Full investigation in `research_log/VALIDATION.md` VAL-012 and `models/canary-qwen/docs/PROGRESS.md`.

Decision: Drop the research-optimized ablation entirely from the active manuscript, research reports (`research_report/*.md`), and observational docs (`shared/model_comparison.md`, `PROGRESS.md`). It is not part of the canonical v1/v2/v3 comparison going forward.

Reason: User instruction — the ambiguity (two candidate configs, one of which never even converged) makes this citation weak evidence regardless of which config is "correct," and re-deriving it would cost real GPU-hours to resolve a citation that isn't central to the paper's argument.

Consequences: `shared/model_comparison.md`'s hyperparameter-ablation table, `PROGRESS.md`'s run history, and `research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md`'s evidence table were all updated to remove or clearly flag this row as dropped, rather than silently deleting the historical record — the original citation and its ambiguity are preserved in `PROGRESS.md`'s history for the audit trail.

Rejected Alternatives: Retraining one or both variants to resolve the ambiguity was considered but rejected — not worth the GPU-hours for a data point that isn't part of the core research question ([[DEC-005]]).

Related Records: [[VAL-012]], [[DEC-005]], [[EXP-011]]

---

## DEC-008 — Rename Canary-Qwen UWB-ATCC configs/scripts/results to a canonical v1/v2/v3 scheme

Date: 2026-09-08
Status: DECIDED and EXECUTED (user instruction)

Question: The Canary-Qwen UWB-ATCC config/script/result naming was inconsistent and confusing (`salm_uwb_atcc.yaml` for the baseline, `salm_uwb_atcc_unfrozen.yaml` for encoder-unfrozen, `salm_uwb_atcc_v3.yaml` for the regularized run — no clean numbering) and directly contributed to the confusion this session about which model produced which result. Should the repo be renamed to a consistent scheme?

Decision: Rename to a canonical v1 (LoRA baseline) / v2 (encoder unfrozen) / v3 (LoRA + regularization) scheme across configs, wrapper scripts, and result JSONs:
- `salm_uwb_atcc.yaml` → `salm_uwb_atcc_v1.yaml`; `salm_uwb_atcc_unfrozen.yaml` → `salm_uwb_atcc_v2.yaml`; `salm_uwb_atcc_v3.yaml` unchanged.
- `train_canary_lora.sh` → `train_canary_v1.sh`; `train_canary_unfrozen.sh` → `train_canary_v2.sh`; new `train_canary_v3.sh` added (didn't exist before).
- `finetuned_results_v2.json` (was actually v1/baseline data) → `finetuned_results_v1.json`; `finetuned_results_unfrozen.json` → `finetuned_results_v2.json`; `v3_results.json` → `finetuned_results_v3.json` (for naming consistency). Same pattern for `learning_curve_*.json`.
- All renames done via `git mv` to preserve history.

Reason: User instruction, directly motivated by this session's confusion (the old "v2" name was already in use for something unrelated to actual model versioning, contributing to a caching-bug-driven misunderstanding — see [[VAL-012]]).

Consequences: `REPLICATION_GUIDE.md` and `models/canary-qwen/docs/PROGRESS.md` updated to reference new filenames. The stale `params_trained_pct: 32.8` value in the renamed v2 results JSON was also corrected to 29.2 while it was being touched (long-standing [[ISS-004]] gap, now closed for this file).

Related Records: [[VAL-012]], [[ISS-004]], [[ISS-007]]
