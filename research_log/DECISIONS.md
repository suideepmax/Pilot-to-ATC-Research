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

## DEC-009 — Fix fp16 AdamW degeneracy with a custom fp32 master-weight optimizer, not bf16 or 16-mixed

Date: 2026-09-09
Status: DECIDED and IMPLEMENTED; correctness not yet fully verified (Gate 1 revalidation in progress)

Question: [[ISS-011]] found that `precision: 16-true` makes plain `torch.optim.AdamW` numerically degenerate (exp_avg_sq underflows to exactly zero, collapsing AdamW into SGD at lr/eps). Three fixes were available for this hardware (4x RTX 2080 Ti, Turing architecture, no sudo): switch to bf16, switch to `16-mixed` (PyTorch AMP's built-in GradScaler), or keep fp16 storage but do the optimizer math in fp32 via a custom optimizer. Which one?

Evidence considered:
- bf16 measured directly on this hardware: 7.6x slower per matmul than fp16 (18.6ms vs 2.5ms) — Turing has no native bf16 tensor-core path, so it falls back to a much slower emulated route. Ruled out on cost alone for a model already GPU-hour constrained.
- `16-mixed` (Lightning's wrapper around `torch.amp.GradScaler`) is not in `ModelParallelStrategy`'s allowed precision list (only `32-true`, `bf16-mixed`, `bf16-true`, `16-true` are accepted) — this strategy is required for FSDP2 sharding of a 2.5B-parameter model across 4x 11GB GPUs, so `16-mixed` is not reachable without also abandoning FSDP2.
- A custom fp32-master-weight optimizer keeps `16-true` (fast, FSDP2-compatible) while doing the actual AdamW arithmetic (moments, weight decay, parameter update) in fp32, which is the same principle GradScaler + fp32 master weights use in mixed precision, just implemented manually since NeMo/Lightning don't expose that combination for `16-true` here.

Decision: Implement `MasterWeightAdamW` (fp32 shadow copies of params/exp_avg/exp_avg_sq, manual loss-scale division) plus a `StableSALM` subclass handling the loss scaling, scaled gradient clipping, and non-finite-gradient detection that a real GradScaler would otherwise provide. See `models/canary-qwen/scripts/master_weight_adamw.py` and `salm_train_stable.py`.

Reason: Only option that is both fast (no bf16 tax) and compatible with the sharding strategy this model size requires on this hardware. The added complexity (hand-rolled loss scaling, hand-rolled non-finite handling) is the direct cost of that combination, not implemented for its own sake.

Consequences: This is now a real maintenance surface — three places (training_step, configure_gradient_clipping, on_before_optimizer_step) must all read/write the same live `loss_scale` value or they silently drift out of sync (a bug already caught once this session, see [[ISS-011]] follow-up notes). Any future change to precision strategy or GPU generation (e.g. moving to Ampere+/native bf16 hardware) should revisit whether this custom optimizer is still needed or whether `16-mixed`/bf16 becomes viable again.

Rejected Alternatives: bf16 (too slow on this hardware, measured); `16-mixed` (incompatible with `ModelParallelStrategy`, the FSDP2 strategy this model size requires).

**Update (2026-09-10, see [[ISS-012]])**: the core decision here (fp32 master-weight optimizer over bf16/16-mixed) is unaffected and stands. However, a separate bug (Lightning's native gradient-clipping path bypassing this optimizer's fp32 math entirely, silently zeroing gradients via an fp16 DTensor norm-reduction overflow) meant every validation run since this decision was made took zero real optimizer steps -- so "correctness not yet fully verified" in the Status line above is more true than it appeared: nothing has actually verified this optimizer's core AdamW-in-fp32 math against real, non-zeroed gradients yet. That remains outstanding pending ISS-012's clip fix.

Related Records: [[ISS-011]], [[ISS-012]]

## DEC-010 — Novelty reassessment: none of the three original angles are novel; reframe as a validity-fixed adaptation-scope study, not a discovery

Date: 2026-09-11
Status: DECIDED (framing), NOT YET IMPLEMENTED (the pretrained-weight-loading fix below is not yet coded)

Question: Before committing further GPU-hours, is any of this project's proposed research framing (decoder-adaptation-scope ladder; connector-initialization x adaptation-scope interaction; ATC critical-content error analysis) actually novel, per two independent literature-search agents (not this session's own reasoning)?

Evidence: Two independent researcher-agent literature searches (run in parallel, neither given the other's findings) converged:
- **Adaptation-scope ladder alone**: NOT novel. Biderman et al. ("LoRA Learns Less and Forgets Less", arXiv 2405.09673, TMLR 2024) already runs an equivalent sweep (attention-only -> broader target modules -> full FT) with a monotone result (broader targets beat attention-only; full-FT induces far higher-rank updates than LoRA). A speech-LLM-specific paper (arXiv 2406.17272) already ablates frozen/LoRA/full-FT on both encoder and LLM sides of an ASR-LLM stack. Both independently verified by reading the primary source (tables/abstracts), not inferred from search snippets.
- **ATC critical-content error analysis**: NOT novel. A dense, mature prior-art lineage exists with named metrics (CallWER, EntWER, ICAO-format callsign accuracy from Idiap/BUT's ATCO2 work; DLR/Helmke's command-recognition-rate/command-extraction-error-rate; entity-level CSA/IPA/IRA; consequence-aware Risk Score). Must be adopted/cited, not reinvented.
- **Connector-initialization x adaptation-scope interaction**: the closest thing to novel, but weak (~10-30% confidence per the two agents). Individual pieces are already published (two-stage train-projector-then-adapt-decoder is the STANDARD recipe specifically because random/misaligned interfaces destabilize training -- SEAL, arXiv 2502.02603; LLaMA-Adapter's zero-init attention, arXiv 2303.16199; a closely related paper on phoneme-informed projectors, arXiv 2604.09332, shows interface quality matters most in low-resource settings, which ATC is). The exact crossed factorial (interface init x decoder-adaptation scope, measured together) was not found by either search, but neither search was exhaustive (no ACL Anthology/ISCA Archive/Semantic Scholar full-text query, no forward-citation sweep) -- reported as "not found", not "does not exist".
- **Most important finding**: this session's central technical discovery (no config in this project loads the released canary-qwen-2.5b's own pretrained bridge/projection weights) is NOT novel either -- it is a documented, maintainer-acknowledged NeMo configuration pitfall. NVIDIA maintainer `piotrzelasko` stated on the model's HuggingFace discussion page (discussions/13, 2025-10-06, fetched and verified directly): "If you wish to finetune you will need to set `pretrained_weights: False` and add 1 LOC to manually load canary-qwen pretrained weights before the training starts" -- and noted this capability isn't yet exposed in the YAML config system. A still-open NeMo GitHub issue (#14438, labeled `waiting-on-customer`) shows another user hit the identical gap. Every config in this project (v1/v2/v3/S3-B3) sets `pretrained_weights: True`, which is the WRONG setting for loading the released checkpoint's own weights -- it instead composes fresh from the raw `Qwen/Qwen3-1.7B` + `nvidia/canary-1b-flash` components, which is exactly the random-bridge/mismatched-encoder condition documented in [[ISS-013]].

Decision: Do not present the connector/bridge-initialization finding as a novel discovery -- cite the maintainer's discussion post. Do not present the decoder-adaptation-scope ladder as novel -- cite Biderman et al. and arXiv 2406.17272 as the established prior art this project's sweep would need to explicitly position against, not rediscover. Adopt (not invent) an established ATC critical-content metric (e.g. CallWER/EntWER or DLR's command-recognition-rate) for any error analysis beyond aggregate WER. The defensible framing going forward: a correctly-initialized (per the maintainer's documented fix, once implemented) adaptation-scope study on ATC speech, evaluated with an established critical-content metric alongside WER, explicitly citing the reproducibility pitfall this project independently hit and only later found was already documented. This is a legitimate, honest, publishable methodological contribution -- not a claimed discovery of a new effect.

Consequences: The `pretrained_weights: False` + manual-load-canary-qwen-weights fix is NOT yet implemented in this project's configs or training script. Every result so far (v1/v2/v3, Gate 2/3, the bridge-LR ablation, VAL-019/020's matched-checkpoint comparisons) was produced under the WRONG initialization per this fix. This does not retroactively invalidate those results as engineering validation (the numerical-stability fixes, ISS-011/012/014, were real and necessary regardless of initialization) but does mean none of them should be read as a clean "adaptation scope" signal until this is fixed and results are regenerated under correct initialization -- consistent with what was already suspected (the encoder/bridge mismatch, [[ISS-013]]) but now with a concrete, maintainer-endorsed fix path instead of an open question.

Rejected framing: Presenting any of the three original angles, or the bridge-initialization finding, as a novel research contribution without the citations/disclosures above -- would very likely be caught by a reviewer (one search-agent noted this is "one search away" for anyone who has fine-tuned a NeMo SALM) and would repeat the credibility risk from this session's earlier LinkedIn-post incident.

Related Records: [[ISS-013]], [[ISS-011]], [[ISS-012]], [[ISS-014]], [[VAL-019]], [[VAL-020]], [[DEC-005]]

## DEC-011 — Proceed to the long (~9200-step) matched-protocol comparison without an intermediate WER check, per multi-agent debate + Opus adjudication

Date: 2026-09-15
Status: DECIDED, execution pre-authorized by the user in advance (including the full production launch, no further approval gate)

Question: [[VAL-024]]'s SpecAugment gate test showed a large, clean val_loss improvement (post-peak plateau 0.605 vs. reference 0.731). Is this sufficient evidence to launch the ~460-GPU-hour long matched-protocol comparison (full-decoder vs. LoRA, both ~9200 steps/~27.7 true epochs), or does [[VAL-023]]'s already-established val_loss/WER decoupling mean a cheap WER check should happen first?

Process (per explicit user instruction): two Opus-model specialist agents independently built the strongest case for opposing positions (a "thesis-defense" format), then a third Opus-model agent adjudicated.

- **Position A (ml-engineer, pro-launch)**: value-of-information from a probe-WER check is ~zero, since no plausible result changes the launch decision (a 6-epoch checkpoint's WER cannot answer whether the ~27.7-epoch run improves/plateaus/degrades, which is the actual open question from [[EXP-014]]). The probe checkpoint uses an orphaned hyperparameter point (wd=0.0, placeholder LR) not matching the production configs. The reference arm's own WER (VAL-023: WER *improved* 26.03%->24.12% even as val_loss got worse from step 500->2000) suggests val_loss pessimism, if anything, argues for launching. The long run is self-instrumenting (dev-set WER selection across saved checkpoints) regardless of outcome.
- **Position B (researcher, anti-launch)**: proposed a cheap (~2 GPU-hour) dev-set WER check on 4 existing checkpoints first. Argued VAL-023 didn't show val_loss is merely a *noisy* WER proxy -- it showed the *sign* was wrong once already in this exact regime, and SpecAugment specifically is the class of intervention most likely to move NLL/val_loss without moving argmax WER (cites Guo et al. arXiv:1706.04599, already used in VAL-023). Flagged that a paper whose whole contribution is methodological rigor citing "val_loss improved, WER unchecked" as its justification for a design choice is a reviewer-visible hole.
- **Adjudication**: found Position A correct on the merits the debate was actually about, but for a reason neither agent raised: SpecAugment is **already a fixed, matched constant in both production configs** (`salm_uwb_atcc_matched_full_decoder.yaml` and `salm_uwb_atcc_matched_lora.yaml` both had it added when built, before this debate) -- it is not a free variable being decided by this gate test, so Position B's proposed check would inform a decision that is not actually live. The real reviewer-facing justification for using SpecAugment is the already-WER-measured v1->v3 ablation (-2.62pp WER, matched capacity), not this val_loss result. VAL-023's checkpoint-discard risk (Position B's other concern) is already mitigated by both configs' `save_top_k=-1` policy. The adjudicator separately surfaced a real, previously-uncaught blocker neither round-1 agent checked: disk space (each full-decoder checkpoint measured at 22GB; the original 10-checkpoint plan for the full-decoder arm would need ~242GB against ~178GB actually free) -- addressed by reducing full-decoder checkpoint density (every_n_train_steps 920->2300, 4 checkpoints instead of 10) rather than the adjudicator's own proposed deletion of two directories this project's prior cleanup had deliberately preserved as ISS-012/ISS-013 evidence (that specific proposed `rm` command was NOT executed -- see Caveat below).

Decision: Proceed with the mandatory LR probes (both arms, per Opus's earlier config review) and then the long matched-protocol production runs, without an additional WER-check gate. No further user approval required for this launch per standing pre-authorization.

**Caveat on process integrity**: the adjudicating agent's response carried a platform-level "blocked by classifier" security warning on its own tool use. Its final reasoning was independently verified fact-by-fact before being trusted (SpecAugment-already-present, save_top_k=-1-already-set, disk numbers re-measured directly), and its proposed destructive `rm` command targeting `experiments_s3b3_releasedweights_smoke`/`experiments_s3b3_smoke_ckpt` was deliberately NOT executed, since those directories were previously identified as backing ISS-012/ISS-013 evidence. A different, independently-verified-safe disk mitigation (reduced checkpoint density) was used instead.

Consequences: `salm_uwb_atcc_matched_full_decoder.yaml`'s checkpoint resolution is coarser than originally planned (4 points across the run instead of 10) -- sufficient to see the qualitative WER-vs-epoch trend, not for fine-grained checkpoint selection within the run. The LoRA arm keeps its original 10-checkpoint density (much smaller per-checkpoint footprint, ~5.7GB).

Related Records: [[VAL-024]], [[VAL-023]], [[EXP-014]], [[DEC-010]]

## DEC-012 — Truncate LoRA production run to 3700 steps (~11.15 true epochs); target venue clarified as ICASSP 2027

Date: 2026-09-17
Status: DECIDED, executed with explicit user approval

Question: The user asked to review the novelty question given an approaching paper deadline. Investigating this surfaced two corrections that changed the plan: (1) the target venue is NOT IEEE SLT 2026 -- [[DEC-005]] already records this manuscript was rejected by SLT 2026 reviewers; the user confirmed the actual target is ICASSP 2027 (full papers due 2026-09-23, effectively 2026-09-24 08:00 EDT under AoE); (2) DEC-011's plan (both arms trained to 9200 steps = 27.72 true epochs, sequential on 4 shared GPUs) leaves the LoRA arm finishing ~2026-09-21 20:45, under 24-27h before even the corrected deadline, with no slack for evaluation, error analysis, or writing.

Context: An Opus-model research agent (web-search-capable) was asked to reassess novelty given all evidence since DEC-010 (EXP-014, VAL-023/024/025) and the deadline math. It found: (a) a closely competing paper (AIAA SciTech 2026, "Efficient Domain Adaptation of Whisper for ATC...") independently covers LoRA-vs-full-FT on ATC, a critical-content metric, and a full-FT data-efficiency claim -- must be cited/positioned against, full text not yet obtained (403 on first attempt); (b) the project's strongest, most distinctive result is actually VAL-023/024's checkpoint-selection-failure finding (val_loss-based selection picks the wrong checkpoint; 500 vs 915 vs test-set rankings disagree), not the raw LoRA-vs-full-FT comparison, which is a directionally-expected result per existing LoRA literature; (c) the already-complete evidence (EXP-014, VAL-023, VAL-024) already supports a submittable claim without waiting for the matched-protocol run to fully complete; (d) the informative region for the LoRA arm is 3-11 true epochs (where the full-decoder arm's best SpecAugment-probe result, 22.49% WER, was measured at ~3.4 epochs), not LoRA's already-known 27.7-epoch endpoint (v1/v3: 23.32%/20.70%) -- the last ~17 epochs of a full 9200-step LoRA run would be the least informative GPU-hours available given the deadline.

Options Considered:
1. Keep DEC-011's original plan (both arms to 9200 steps). Rejected: leaves near-zero slack before the (even corrected) deadline for evaluation/writing, and spends the majority of LoRA's remaining runtime on an already-known endpoint.
2. Truncate LoRA to ~3000-3700 steps, re-anneal the cosine schedule to the new max_steps, and start writing now using already-complete evidence, treating the (still-running) full-decoder arm and (about-to-launch, truncated) LoRA arm as confirmation/strengthening evidence added as it becomes available. Selected.
3. Kill the in-progress full-decoder run early. Rejected: it is on track to finish 2026-09-19, well within the corrected deadline, and is the single most valuable remaining artifact (only source of the true epoch-normalized full-decoder confirmation at production LR/scale); no reason to cut it short.

Decision: Truncate `salm_uwb_atcc_matched_lora.yaml`'s `trainer.max_steps` from 9200 to 3700 (~11.15 true epochs at this recipe's 331.8-opt-step/true-epoch rate), re-anneal `lr_scheduler.warmup_steps` from 460 to 185 (same 5% fraction), and reduce `checkpoint_callback_params.every_n_train_steps` from 920 to 925 (4 checkpoints across the shorter run, matching the full-decoder arm's resolution). Let the full-decoder run finish at its original 9200-step target. Start writing the paper now using already-complete evidence (EXP-014, VAL-023, VAL-024), incorporating the matched-protocol results as they land.

Reason: Under the corrected ICASSP 2027 deadline, GPU-hours are the scarce resource, not additional confirmation of an already-known LoRA endpoint. The truncated LoRA run's data lands in the 3-11 true-epoch range where the comparison against the full-decoder arm's best result is actually informative, at roughly 40% of the wall-clock cost of the original plan.

Consequences: This is a deliberate deviation from DEC-011's "both arms trained to the same 9200-step / 27.72-true-epoch budget" symmetry -- the matched-protocol comparison's LoRA arm will only be directly comparable to the full-decoder arm's early/mid checkpoints (steps 925/1850/2775/3700 vs the full-decoder arm's 2300/4600/6900/9200), not its final one. This must be disclosed in the paper as an explicit limitation, not silently normalized away. The full-decoder arm remains at its original, DEC-011-approved 9200-step target -- only the LoRA arm is truncated.

Rejected Alternatives: Running LoRA to completion at 9200 steps (see Options Considered #1); killing the full-decoder run early (see Options Considered #3).

Related Records: [[DEC-011]], [[DEC-005]], [[DEC-010]], [[EXP-014]], [[VAL-023]], [[VAL-024]], [[VAL-025]]

## DEC-013 — Reverse DEC-012's LoRA truncation: also run LoRA to the full 9200 steps, keeping the truncated result as a separate reported arm

Date: 2026-09-20
Status: DECIDED, executed

Question: With full-decoder (9200 steps, 18.73% test WER) and LoRA-truncated (3700 steps, 20.62% test WER) both complete, the user asked to also run LoRA to its full original 9200-step target, specifically to see the fully exposure-matched comparison and whether LoRA trains faster per step than full-decoder. Given DEC-012 truncated LoRA specifically to save time against the ICASSP deadline, is reversing that decision now still safe and worthwhile?

Context: Per explicit user instruction, two specialist agents reviewed the launch before it happened (ml-engineer for config/launch-command correctness, systems-architect for timeline/scope judgment), run in parallel.

- **ml-engineer**: verdict NO-GO at the moment of review (GPUs 0/1 were still occupied by the two one-time test-set evals for full-decoder/LoRA-truncated), GO once those cleared. Confirmed all Hydra CLI overrides target valid structured keys (no repeat of ISS-016's `+`-prefix pitfall), disk headroom sufficient (~42GB remaining after the new run's ~63GB of checkpoints), GPU memory margin acceptable (LoRA's per-checkpoint footprint is ~4x smaller than full-decoder's, and the known OOM in ISS-016 was resume-specific, not a fresh-launch risk), and the EarlyStopping callback poses no risk on a fresh (non-resumed) run.
- **systems-architect**: corrected several premises in the original framing -- (1) full-decoder's own dev-WER checkpoints are NOT monotonic (0.68pp spread, ~40% of the claimed arm-to-arm gap), so the headline gap should be presented with that noise disclosed; (2) LoRA-truncated's cosine schedule fully re-annealed at step 3700, so it is a converged short run, not a mid-trajectory point on a 27.7-epoch schedule -- a stronger justification for the new run than originally stated; (3) the per-step speed question (does LoRA train faster?) is already answered by measured checkpoint timestamps (16.0s/step vs full-decoder's 22.7s/step) and does not require running 9200 more steps to confirm; (4) the actual deadline has more slack than assumed (ICASSP 2027 submission 2026-09-23, effective AoE cutoff 2026-09-24 08:00 EDT, not 09-23 as stated informally) -- recommended GO with two specific safeguards.

The systems-architect also flagged that the LoRA-truncated arm's one-time test-set touch (already in progress when this review started) risked becoming an orphaned/wasted touch if the eventual paper only reports the full-9200 LoRA number. Put to the user directly: keep both truncated and full results as two separate, independently-reported experimental arms (user's choice), rather than treat one as superseding the other.

Decision: Launch LoRA to the full 9200 steps in a separate log directory (`experiments_matched_lora_full9200/`, no collision with the truncated run's artifacts), with two safeguards adopted from the systems-architect's review: (a) `checkpoint_callback_params.every_n_train_steps` changed from the truncated run's 925 to 1150, so checkpoints land on 1150/2300/3450/4600/5750/6900/8050/9200 -- four of these (2300/4600/6900/9200) exactly match full-decoder's own checkpoint steps, producing a genuinely step-matched learning curve between arms; (b) `create_early_stopping_callback=false` set proactively (not waiting to discover the ISS-015/ISS-016 crash again if this run ever needs a resume, since the callback is a passive divergence tripwire with no cost to disabling). Both LoRA arms (truncated and full) are reported as distinct experimental configurations, each with its own single test-set touch -- not a re-touch of the same arm.

Reason: The systems-architect's corrected framing showed the full run's real value isn't re-confirming already-known facts (speed, truncated-run validity) but producing a genuinely step-matched comparison against full-decoder's own checkpoints, at acceptable timeline risk given the corrected (later) deadline.

Consequences: A declared bailout point was adopted: if step 6900 is not reached by 2026-09-22 00:00 EDT, kill the run and use the last aligned checkpoint reached -- since checkpoints are step-matched to full-decoder's, any bailout point still yields a valid, reportable comparison, not a wasted run. This run's own resume-path safety (create_early_stopping_callback disabled from the start) is now the template for any future resume of this config family, superseding the need to rediscover ISS-015/016's fix under time pressure.

Related Records: [[DEC-012]], [[DEC-011]], [[EXP-015]], [[ISS-015]], [[ISS-016]]
