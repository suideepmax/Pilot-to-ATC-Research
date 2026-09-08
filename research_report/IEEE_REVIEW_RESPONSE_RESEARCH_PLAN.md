# IEEE SLT 2026 Review Response — Research Program Redesign

Status: PLANNING ONLY. No training launched. No source code, scripts, or existing results modified. Nothing committed or pushed.

All facts below marked **[VERIFIED]** were confirmed by direct inspection in this session (file reads, checksums, computed durations, installed package versions, and — after an explicit reconciliation pass — the submitted manuscript itself). Facts marked **[FROM PRIOR RECORD]** come from `research_log/` entries created in earlier sessions/audits — treated as evidence, not fact, per engineering-memory policy. Claims are additionally tagged **VERIFIED FACT** / **INFERENCE** / **OPEN QUESTION** / **HISTORICAL CLAIM WITH UNCERTAIN PROVENANCE** where the distinction matters. Everything else is explicitly labeled as a **recommendation** or **open question**.

**Reconciliation note (added after reading the submitted manuscript, `Final_Draft.md`):** an initial draft of this report was written before the manuscript itself was available, using only the reviews, repo, and configs. Reading the manuscript surfaced corrections to that draft — most importantly, the "13h vs 10.5h" figure is **not** a data error (both numbers are correct once you know one is train-only and one is train+test — see Section 2, item 12), and the encoder-unfrozen ablation's status is more nuanced than "invalid" (see Section 3.G and 4). Conflicts between the manuscript, the repo, and engineering-memory are shown explicitly below rather than silently resolved, per instruction.

---

## 1. Executive Summary

All four reviewers rejected the paper (rating 2/5 each), and three of four independently converged on the same core complaint: **the comparison between Wav2Vec2 and Canary-Qwen cannot isolate what caused the observed gap**, because architecture, scale, pretraining, adaptation scope, and decoding all differ simultaneously. A fourth, cross-cutting complaint is that **"unseen domain" is factually wrong** — every model is fine-tuned and tested on the same corpus's own held-out split.

Investigation of the actual repo this session surfaced two findings the reviewers didn't even know to ask about, which make the situation worse than the reviews describe:

- **The "encoder unfrozen" Canary-Qwen config committed to this repo is byte-identical (MD5-verified) to the frozen baseline config**, yet the manuscript reports a distinct 838.8M-param/23.82%-WER result with its own learning-curve trajectory (Table III). Checkpoints are lost, so neither the manuscript nor the repo config can be independently re-verified against ground truth. **Classification: HISTORICAL CLAIM WITH UNCERTAIN PROVENANCE, not simply invalid** — the WER trajectory's internal plausibility makes it more likely the repo config is the stale artifact than that the result was fabricated, but this is a judgment call, not a verified fact (Section 3.G).
- **The "v3 regularization" config changes three variables at once** (`lora_dropout` 0.01→0.1, adds SpecAugment, `weight_decay` 1e-3→1e-2) — confirmed by diff, exactly matching Reviewer BzrK's and Reviewer CXTt's confounding complaint, and exactly matching the manuscript's own description (Sec. IV.C) of what changed between the original run and v3.

Given you've lost the original checkpoints/data anyway, this is not a "patch the paper" situation — it's a from-scratch retraining program. The right response is to **narrow the paper's claim to something the available data, hardware, and time can actually support**, not to try to satisfy every reviewer request. Section 5 recommends the single strongest reframing: a **controlled decoder-adaptation-scope study on Canary-Qwen, paired with a properly ablated regularization study and a corrected speaker-independent ATCOSIM evaluation** — dropping the "unseen domain" framing and the architecture-vs-architecture headline claim entirely.

---

## 2. Reviewer Concern Matrix

| # | Concern | Reviewer(s) | Classification | Why |
|---|---|---|---|---|
| 1 | Apples-to-oranges LM comparison (W2V2+KenLM vs Canary no-LM) | 4Pd7 | **CRITICAL SCIENTIFIC** | Directly invalidates the paper's central claim ("W2V2 is better") — the comparison is contaminated by an unmatched decoding condition, not just an architecture difference. |
| 2 | Decoder-LoRA / adaptation-scope conclusions unsupported (frozen-decoder hypothesis untested) | 4Pd7, CXTt | **CRITICAL SCIENTIFIC** | The paper's own interpretation (Sec. VI.B/F) rests on an experiment that wasn't run — this is a logical gap in the argument itself, not a presentation issue. |
| 3 | Regularization conclusion confounded (3 variables changed at once) | CXTt, BzrK | **CRITICAL SCIENTIFIC** | **[VERIFIED]** Confirmed by diffing the actual configs — `lora_dropout`, SpecAugment, and `weight_decay` all changed together. The causal claim in the paper is not supported by the experiment that was run. |
| 4 | "Unseen domain" is inaccurate (same-corpus 80/20 split) | X1FA, CXTt, BzrK | **CRITICAL SCIENTIFIC** | Three of four reviewers independently flagged this — it's a terminology/methodology error that undermines the title and framing, not a wording nitpick. |
| 5 | Uncontrolled architectural/scale/pretraining differences prevent isolating full-FT vs LoRA | X1FA, CXTt, BzrK | **CRITICAL SCIENTIFIC** | This is the meta-complaint underlying #1–3: the experimental design cannot answer the question the title poses. |
| 6 | Limited scientific novelty (result direction is expected) | 4Pd7, X1FA, CXTt, BzrK | **HIGH PRIORITY** | Universal across all four reviews. Can't be fixed by more experiments alone — requires reframing the contribution (Section 6). |
| 7 | No data-scale analysis (how much data would Canary need to win?) | CXTt | **HIGH PRIORITY** | A concrete, answerable question reviewers explicitly proposed — directly actionable and would address novelty concern #6 simultaneously. |
| 8 | No speaker-independent evaluation | X1FA, CXTt, BzrK | **HIGH PRIORITY** | You already have this partially underway (EXP-007, this repo) — closing this gap is now the cheapest of the major asks since the work is substantially done. |
| 9 | No cross-corpus evaluation | X1FA, CXTt, BzrK | **HIGH PRIORITY** | Reviewers treat this as an alternative (not additional) way to earn the "unseen domain" claim — see Section 10 for feasibility verdict. |
| 10 | Training-time diagnostics over-emphasized vs. held-out results | 4Pd7 | **MEDIUM PRIORITY** | A presentation/rigor issue, but it also reveals the paper leans on training curves in place of a controlled test-set comparison — partially a symptom of #5. |
| 11 | No figures — learning curves shown as tables | 4Pd7 | **PRESENTATION ONLY** | Real and fixable in an afternoon once final numbers exist; doesn't change scientific validity. |
| 12 | Data-scale figures inconsistent (13h vs 10.5h) | 4Pd7 | **PRESENTATION ONLY** | **RESOLVED, not an error — [VERIFIED FACT]** computed directly from `segments` files: UWB-ATCC train = 10.54h, test = 2.63h, **train+test = 13.17h**. Sec III.B's "13 hours" (train+test) and Sec VI.B's "10.5 hours" (train only) are **both numerically correct** — the manuscript just never labels which subset each figure refers to. My initial pre-manuscript draft of this report incorrectly called the 13h figure "unsupported/wrong"; that claim is retracted after reading the actual text. Fix: label each figure explicitly as train-only or train+test in the redesign. |
| 13 | Reproducibility/documentation gaps, typos, section ordering, template title | 4Pd7 | **PRESENTATION ONLY** | Mechanical fixes with zero research content. |
| 14 | Discussion lists results without deeper interpretation | X1FA | **MEDIUM PRIORITY** | Consequence of #5/#6 — once the research question is narrower and better controlled, the interpretation will have somewhere real to go. |
| 15 | Need to justify which experiments are scientifically informative | CXTt (implicit), all | **HIGH PRIORITY** | This is the organizing principle for Sections 4/16 below — avoid the reviewers' own trap of over-experimenting without resolving the actual question. |
| 16 | Comparing Canary-Qwen "as a bigger model" is unfair given most of its params serve language modeling, not acoustic recognition | BzrK | **MEDIUM PRIORITY** | A framing critique more than an experimental one — addressed by reframing the contribution (Section 6), not by new training. |
| 17 | Missing comparison to other architectures (e.g., Whisper) | BzrK | **OPTIONAL** | Explicitly de-scoped: adding a third model family multiplies the experimental matrix without resolving any of the CRITICAL items above. Correctly deferred to future work. |

---

## 3. Current Research-State Reconstruction

**A. What was actually reproduced from upstream (idiap/w2v2-air-traffic)?** **[VERIFIED]** The full W2V2-large fine-tuning pipeline (`src/run_asr_fine_tuning.sh`, `src/run_speech_recognition_ctc.py`, `src/eval_model.py`, `src/train_kenlm.py`) is the upstream code, cloned directly from `github.com/idiap/w2v2-air-traffic` (confirmed: that repo's `origin` remote points there). The UWB-ATCC and ATCOSIM data-prep pipelines and paper hyperparameters (lr, mask_time_prob, warmup) are also upstream-derived.

**B. What was changed for the available hardware?** **[FROM PRIOR RECORD, DEC-003]** `python3` (DataParallel) → `torchrun --nproc_per_node=4` (DDP), because DataParallel OOMs an 317M-param model on 11GB cards. `per_device_train_batch_size` reduced from the paper's 16/24 to 1, with `gradient_accumulation` raised to compensate (effective batch 64 vs. paper's 96–128). `fp16_full_eval=False` to avoid a cuBLAS crash. `min_duration_in_seconds` raised from 0.2→0.5 to avoid a mask-length crash on short clips. AdamW `eps` raised to 1e-4 for Canary-Qwen (NeMo/FSDP) to avoid fp16 NaN — **[FROM PRIOR RECORD, ISS-003]**.

**C. Which changes were committed to GitHub?** **[VERIFIED]** The UWB-ATCC-generation scripts (`train_wav2vec2_large.sh`, `train_wav2vec2_base_*.sh`, `eval_large_model.sh`, `train_kenlm.sh`) and Canary-Qwen UWB-ATCC configs (`salm_uwb_atcc.yaml`, `salm_uwb_atcc_unfrozen.yaml`, `salm_uwb_atcc_v3.yaml`) are in `Pilot-to-ATC-Research`. **The ATCOSIM Canary-Qwen configs (`salm_atcosim.yaml`, `salm_atcosim_v3.yaml`) exist only in the working clone `~/w2v2-air-traffic/ablations/atcosim/` and were never pushed to GitHub** — confirmed absent by an exhaustive `find`. Anyone trying to reproduce the ATCOSIM Canary-Qwen numbers from the public repo alone cannot do so; the config simply isn't there.

**D/E. Original replication vs. modified-config results:** Every large-model result in this repo (W2V2-large on both corpora, Canary-Qwen on both corpora) used the hardware-adapted config (item B), not the paper's original config. **No result in this repo was produced under paper-identical settings.** This is disclosed in the docs (`PROGRESS_ATCOSIM.md`'s own hyperparameter table marks every deviation with ✗), which is good practice — but it also means "we beat the paper" comparisons (Sec. "Why we beat the paper") are confounded by LR, batch size, and DDP-vs-DP simultaneously, similar in kind to the v3 regularization confound.

**F. Which results are actually reproducible from the current scripts?** UWB-ATCC W2V2-large and ATCOSIM W2V2-large (baseline + gender splits) are reproducible — the scripts, configs, and data manifests are present and internally consistent (this session's `VAL-004`/`VAL-005` already verified this for the gender-split scripts). **UWB-ATCC Canary-Qwen LoRA and v3 are reproducible** — configs present, diffed, and understood. **ATCOSIM Canary-Qwen (v1, v3) is NOT reproducible from `Pilot-to-ATC-Research` alone** (config missing, item C) — it is reproducible from `w2v2-air-traffic` if that clone is preserved. **The Canary-Qwen "encoder unfrozen" result is NOT reproducible from either location** — see G.

**G. Which historical results are invalid/unreliable?**
- **ATCOSIM gender-eval leaked results (0.86%/0.01%)** — already known-invalid, `[FROM PRIOR RECORD, ISS-001]`, correctly superseded this session by EXP-007 (female 4.85%, male 19.97%, `VAL-006`/`VAL-007`), which also directly fulfills the manuscript's own stated future work (Sec. VI.E: *"testing on held-out speakers... remains future work"*) — **VERIFIED FACT**, checked against the manuscript text directly.
- **Canary-Qwen "encoder unfrozen" ablation (WER 23.82%, 838.8M/29.2% trainable)** — **RECLASSIFIED after reading the manuscript: HISTORICAL CLAIM WITH UNCERTAIN PROVENANCE, not simply invalid.** `salm_uwb_atcc_unfrozen.yaml` and `salm_uwb_atcc.yaml` are MD5-identical (**VERIFIED FACT**); both currently freeze `^perception\\.encoder\\..+$`. But the manuscript reports a specific, internally-consistent learning-curve trajectory for this run (Table III: 46.34%/57.37% at steps 500/1000, distinct from and initially worse than the LoRA-only curve, converging to a plausible 23.82% by step 10,000) — a level of detail that argues against outright fabrication. Checkpoints are lost (per your own statement), so neither the manuscript's claim nor the repo's config can be independently verified against ground truth. **My judgment** (stated as judgment, not fact): more likely the committed YAML is a stale/overwritten artifact than that the paper's numbers are fake — but this is not resolved, and **this result cannot be cited in the redesigned paper without either the authors' independent recollection of the actual config used, or re-running it under a freshly-verified config.**
- **Manuscript states "six training runs were conducted" (Sec. IV.C) for Canary-Qwen, but a direct count of Table I (4 fine-tuned configs) + the 3×10⁻⁵ ablation described only in prose (Sec. VI.C, not tabulated) + ATCOSIM (adapter-only + v3 = 2) totals 7, not 6** — **VERIFIED via direct count against the manuscript.** Minor manuscript-accuracy item, not scientifically consequential; flag for the redesign's cleanup pass.
- **Canary-Qwen v1→v3 "regularization fixed the plateau" conclusion** — not invalid, but **confounded**, confirmed by diff (3 variables changed together). Usable as a *motivating observation*, not as a controlled finding.
- **The existing ATCOSIM 4-gram KenLM** — **[FROM PRIOR RECORD, ISS-005]** trained on the leaked (non-speaker-disjoint) split; cannot be reused for speaker-independent decoding without retraining on `train_female`/`train_male` text only.

**H. Experiments proposed but never executed (before this session):** ATCOSIM Phase 4 (speaker-independent) — **now executed** (EXP-007, this session, female/male WER 4.85%/19.97%, verified). UWB-ATCC W2V2 dropout/mask_time_prob ablations — still not executed.

**I. Experiments with sufficient provenance to reuse as-is:** UWB-ATCC W2V2-large (14.54%/12.69%), ATCOSIM W2V2-large baseline (1.67%/1.28%, **caveat: leaked split, single-corpus comparison point only**), UWB-ATCC Canary-Qwen LoRA (23.32%) and v3 (20.70%) — configs verified present and internally consistent, though v3 is confounded as noted. EXP-007 female/male (this session).

**J. Unsupported assumptions in the current paper:** (i) that the W2V2 vs. Canary-Qwen comparison isolates full-FT vs. LoRA (it doesn't — confirmed by the multi-factor differences in B/C above); (ii) that regularization, not the frozen decoder, explains the v1→v3 gain (confounded, per G); (iii) that "unseen domain" describes the evaluation protocol (it's same-corpus held-out, not unseen); (iv) that the encoder-unfrozen ablation demonstrates decoder-bottleneck (invalid config, per G).

---

## 4. Valid vs. Invalid Historical Evidence — Summary Table

| Result | Status | Reusable As-Is? |
|---|---|---|
| UWB-ATCC W2V2-large (14.54%/12.69%) | VALID | Yes |
| ATCOSIM W2V2-large baseline (1.67%/1.28%) | VALID, but same-speaker split (documented limitation) | Yes, with explicit leakage caveat already in docs |
| ATCOSIM EXP-007 female/male (4.85%/19.97%) | VALID, verified this session | Yes |
| UWB-ATCC Canary-Qwen LoRA (23.32%) | VALID | Yes |
| UWB-ATCC Canary-Qwen v3 (20.70%) | VALID result, INVALID causal interpretation (confounded) | Reuse the number; do not reuse the "regularization is the cause" claim without new ablations |
| UWB-ATCC Canary-Qwen "encoder unfrozen" (23.82%, 29.2%) | **HISTORICAL CLAIM, UNCERTAIN PROVENANCE — repo config does not match manuscript's claim, checkpoints lost so neither can be verified** | **No — must re-run under a verified config, or cite with an explicit provenance caveat, or drop from the redesign** |
| UWB-ATCC Canary-Qwen 3×10⁻⁵/2,500-step "research-optimized" run(s) (60.46% WER) | **DROPPED (2026-09-08, user decision)** — removed from the active manuscript and research record; two ambiguous config variants existed (r=64/4-proj, NaN'd and never evaluable; r=128/2-proj, produced the citation), neither has a surviving checkpoint | No — not part of the redesign |
| ATCOSIM Canary-Qwen v1/v3 (7.06%/3.33%) | VALID result, config only in `w2v2-air-traffic`, not GitHub | Yes, but push the config before claiming reproducibility |
| ATCOSIM gender leaked eval (0.86%/0.01%) | INVALID (known) | No — already superseded |

---

## 5. Revised Research Question

The reviewers are right that "does full-FT beat LoRA in general" is not answerable with two architecturally-unmatched models. Four candidates, evaluated against your actual constraints (4×11GB GPUs, two mid-size ATC corpora, checkpoints lost so full retraining is already required):

**Candidate A — Data-efficiency crossover.** *"How much in-domain data does a LoRA-adapted SALM need to match a fully fine-tuned CTC model on ATC speech?"*
- Scientific value: directly answers Reviewer CXTt's explicit question; genuinely novel framing (crossover point, not just an endpoint comparison).
- Required experiments: W2V2 + Canary-Qwen each at 3–4 data scales (Section 7), both corpora ideally, single adaptation config held fixed.
- Compute: moderate-high (multiplies by data-scale factor).
- Feasibility: high — subsetting the existing manifests is cheap and needs no new code.
- Likely objections: still two different architectures, so the "why" behind any crossover remains partially confounded — but the *data-efficiency curve itself* is the finding, not an architecture-attribution claim, which sidesteps reviewers' core complaint.
- Recommendation: **strong candidate, but expensive if run on both corpora at every stage — see Section 7 for a scoped version.**

**Candidate B — Decoder adaptation-scope study (Canary-Qwen only).** *"Within a fixed SALM architecture (Canary-Qwen), how does adaptation scope (LoRA q/v-only → broader LoRA → full decoder fine-tune) affect ATC WER, and does it close the gap to a fully fine-tuned CTC baseline?"*
- Scientific value: directly tests the exact hypothesis the paper currently asserts without evidence (frozen-decoder bottleneck) — this is Reviewer 4Pd7's Major Concern #2, verbatim.
- Required experiments: 3 adaptation scopes × 1 corpus (or 2, see Section 6), same data/decoding held fixed.
- Compute: low-moderate — reuses the exact training pipeline already built, just changes `freeze_params`/`lora.target_modules`.
- Feasibility: **highest of all four candidates** — the YAML mechanism for this already exists and is understood (Section 6).
- Likely objections: still doesn't make Canary-Qwen and W2V2 comparable; a reviewer could ask "so what if full-decoder-FT closes the gap — you still haven't shown LoRA vs. full-FT under matched conditions." Mitigated by NOT claiming to resolve the cross-architecture question — this experiment's claim is about Canary-Qwen's own adaptation-scope sensitivity, full stop.
- Recommendation: **strongest single candidate** — cheap, directly answers a reviewer-stated gap, and is honestly scoped to what the architecture allows.

**Candidate C — Speaker-independence and cross-corpus generalization framing.** *"How well do a fully fine-tuned CTC model and a LoRA-adapted SALM generalize to held-out speakers and to a different ATC corpus, after in-domain adaptation?"*
- Scientific value: directly answers Reviewers X1FA/CXTt/BzrK's "unseen domain" complaint by replacing it with an honestly-scoped generalization study.
- Required experiments: speaker-independent eval (mostly done — EXP-007) + cross-corpus eval (Section 10 — feasibility is mixed, see below).
- Compute: low for speaker-independence (done), moderate-high for cross-corpus if pursued.
- Feasibility: speaker-independence — high, already largely executed. Cross-corpus — **methodologically weak, see Section 10** — UWB-ATCC (8kHz, real noisy) and ATCOSIM (32kHz→16kHz, clean scripted) differ enough in domain and vocabulary that a naive cross-corpus number risks being uninterpretable rather than illuminating.
- Recommendation: keep the speaker-independence half (cheap, already done, directly closes a reviewer gap); **treat cross-corpus as optional/exploratory, not a headline claim** (Section 10 develops this).

**Candidate D — Compute-efficiency-vs-accuracy tradeoff.** *"Under a fixed GPU-hour budget, does full fine-tuning of a smaller CTC model or parameter-efficient adaptation of a larger SALM deliver better ATC WER per GPU-hour?"*
- Scientific value: reframes the comparison around a practically meaningful axis (compute budget) rather than an apples-to-oranges accuracy claim.
- Required experiments: same runs as the original paper, but analyzed by GPU-hours-to-target-WER rather than final WER alone.
- Compute: no new training beyond what's needed for B, but changes the *analysis*, not the *design*.
- Feasibility: high, cheapest of the four (mostly re-analysis of runs required for other reasons).
- Likely objections: doesn't resolve the underlying "what factor causes the gap" question at all — it's an efficiency framing layered on the same confounded comparison, so alone it doesn't satisfy reviewers' core complaint.
- Recommendation: **use as a secondary/supporting angle within whichever of A/B/C is primary, not as the sole redesign.**

### Recommendation: adopt B as the primary research question, with C's speaker-independence half and D's efficiency framing as secondary supporting analyses.

Rationale: B is the only candidate that (1) is fully executable with existing code and modest compute, (2) directly tests a claim the current paper makes without evidence, (3) is honestly scoped (it doesn't overclaim resolution of the cross-architecture question), and (4) produces a genuinely novel, defensible contribution — a controlled adaptation-scope ablation on a hybrid SALM for a safety-critical low-resource domain, which the reviewers themselves said doesn't exist yet ("Decoder-LoRA... is currently future work").

---

## 6. Scientific Contribution Strategy

**Chosen contribution:** *"A controlled study of adaptation scope in a hybrid Speech-LLM (Canary-Qwen) for air-traffic-control ASR, isolating how much of the performance gap to a fully fine-tuned CTC baseline is attributable to adaptation scope versus architecture — evaluated with speaker-independent held-out testing."*

| Candidate direction | Evidence required | Novelty | Rejection risk | Feasibility |
|---|---|---|---|---|
| Data-efficiency comparison (A) | Multi-scale runs, both models | Medium-high | Medium (still cross-arch) | Medium |
| **Adaptation-scope comparison (B) — CHOSEN** | 3 Canary-Qwen configs, held-out eval | **High** (reviewers confirm this gap exists) | **Low** (directly answers stated concern) | **High** |
| Speaker-shift robustness (C-half) | EXP-007 results + write-up | Medium | Low | Already done |
| Cross-corpus transfer (C-half) | New training + eval | Medium | Medium-high (methodologically contested, Sec. 10) | Low-medium |
| Compute-efficiency tradeoff (D) | Re-analysis only | Low-medium alone | Low | High |
| Full fine-tuning of Canary-Qwen | New training, likely OOM risk on 11GB cards | High if feasible | High (may not be technically feasible — needs verification) | **Unverified — flag as open question, not a plan** |

Do NOT choose "full fine-tuning of Canary-Qwen" as a headline unless Section 6 (below) confirms it fits in 11GB VRAM even with FSDP — this needs a technical feasibility check before committing, not an assumption.

---

## 7. Experimental Matrix (Factors)

| Factor | MUST TEST | SHOULD TEST | OPTIONAL | DO NOT TEST |
|---|---|---|---|---|
| 1. Model/architecture | W2V2-large (fixed baseline), Canary-Qwen (primary subject) | — | — | A third architecture (e.g. Whisper) — correctly out of scope per reviewer BzrK's own comment being classified OPTIONAL |
| 2. Data scale | — | 2–3 subset sizes on ONE corpus (UWB-ATCC, largest/most realistic) | Second corpus at same scales | Full 5-point grid (10/25/50/75/100%) on both corpora — too expensive for the marginal information gained once the crossover trend is visible |
| 3. Adaptation scope (Canary-Qwen) | (a) current baseline: LoRA q/v-only, encoder frozen; (b) LoRA on q/v/k/o + MLP (broader decoder LoRA); (c) genuinely-unfrozen-encoder + LoRA-decoder (re-run, since the old config is invalid) | Full decoder fine-tune, IF VRAM permits (verify first) | — | Exhaustive combinatorial grid of every LoRA target × every freeze combination |
| 4. Regularization (component-wise) | SpecAugment alone; dropout alone; weight_decay alone (each vs. the ORIGINAL v1-equivalent baseline) | Best pairwise combination suggested by single-factor results | Full factorial (8 configs) | Repeating v3's exact 3-way-combined config as if it were an ablation — it isn't one |
| 5. Decoding/LM | W2V2 greedy, W2V2+KenLM (already have this), Canary native, Canary N-best rescored with the SAME KenLM | Canary ILME-style correction, if feasible (Section 9) | — | Naive shallow-fusion of the KenLM directly onto Canary-Qwen's output (the exact mistake reviewers flagged) |
| 6. Speaker generalization | ATCOSIM train_female/male vs. held-out speakers (**already done, EXP-007**) | Per-speaker WER breakdown (gm1 vs gm2 individually — flagged as an open question in `EXPERIMENTS.md` already) | — | Treating pooled 2-speaker WER as a precise generalization estimate without reporting per-speaker variance |
| 7. Cross-corpus | — | — | UWB-ATCC→ATCOSIM only (not the reverse — see Section 10 for why the direction matters) | Both directions as a symmetric claim — the corpora are too different for a naive bidirectional framing to be informative |

---

## 8. Apples-to-Apples Comparison Design

**A. W2V2 + greedy** — already have this (14.54% UWB-ATCC / 1.67% ATCOSIM). Keep as the no-LM anchor.

**B. W2V2 + in-domain KenLM** — already have this (12.69% / 1.28%). Keep as the LM-assisted anchor.

**C. Canary-Qwen + native decoding** — already have this (the current WER numbers). This is Canary-Qwen's own internal LM (the Qwen3 decoder) already doing implicit language modeling — this is NOT directly comparable to W2V2's greedy CTC output, because Canary already has an LM baked in and W2V2's greedy doesn't.

**D. Canary-Qwen + principled external-LM treatment** — this is where the reviewer's Major Concern #1 lives. Two options were proposed:
- **ILME/density-ratio correction**: requires estimating `p_ILM(y)` by running the decoder with the acoustic/encoder contribution removed, then combining `log p_SALM(y|x) + λ·log p_ext(y) − γ·log p_ILM(y)`. **This requires custom decoding-time code that does not exist in the installed NeMo speechlm2 module as far as this session verified** — it is a nontrivial engineering task (estimating a text-only LM score from a model that was never trained without its audio conditioning is not a one-line change). **Open question, flagged for follow-up**: does `nemo.collections.speechlm2`'s `SALM.generate()` expose any hook for audio-free forward passes needed to estimate `p_ILM`? This needs direct inspection of the installed NeMo source before committing to this path — do not assume it's straightforward, per your own instruction.
- **N-best rescoring**: decode Canary-Qwen to an n-best list (requires `num_return_sequences`/beam-search support in `SALM.generate()`), then rescore with the same in-domain KenLM used for W2V2. **This is the lighter-weight option**, but its feasibility depends on whether `SALM.generate()` supports multiple-hypothesis decoding at all — **not yet verified in this session; this is a Stage-0 technical spike (Section 12), not an assumption.**

**E. Canary-Qwen + N-best rescoring** — same as D's second bullet; listed separately per the review structure but functionally the same experiment.

**Recommendation:** N-best rescoring (D/E) is very likely the cheaper, more defensible path IF `SALM.generate()` supports `num_return_sequences` — this must be verified against the actual installed NeMo source (`~/miniconda3/envs/canary_ft/.../nemo/collections/speechlm2/`) before the experiment is scheduled. If N-best generation is not supported, the fallback is to **honestly report Canary-Qwen's native (no-external-LM) WER against W2V2's native (no-external-LM, i.e. greedy) WER as the primary controlled comparison**, and report the KenLM-assisted W2V2 number as a separate, clearly-labeled "with in-domain LM" data point rather than the headline comparison. This avoids the exact contamination Reviewer 4Pd7 flagged without requiring engineering work whose feasibility is currently unverified.

---

## 9. Canary-Qwen Adaptation Study

**[VERIFIED FACT]** from `salm_uwb_atcc.yaml`: the current baseline config already does **decoder-only LoRA restricted to `q_proj`/`v_proj`**, with `freeze_params` covering the LLM base weights, embed_tokens, and the entire perception (encoder+preprocessor) stack. LoRA (via PEFT, `task_type: CAUSAL_LM`) injects new trainable low-rank adapters on top of the frozen `llm` weights regardless of the freeze list — this is standard PEFT behavior.

**Reconciled against the manuscript directly:** Sec. III.C confirms LoRA was already "applied to the LLM decoder and the modality adapter" — matching the verified config exactly. The Conclusion's Future Work section separately lists "LoRA adaptation of the Qwen3-1.7B decoder layers directly" as future work, which reads as contradictory until parsed narrowly: the *existing* LoRA only touches attention q/v projections, so "future work" most likely means extending to more projections (k/o, MLP) or full fine-tuning — not that no decoder adaptation exists. **This ambiguity in the manuscript's own wording, not a gap in my repo investigation, is what led Reviewer 4Pd7 to describe decoder-LoRA as "currently future work."** The redesign must state this precisely: q/v-only decoder LoRA is the existing baseline; broader-scope LoRA and full decoder fine-tuning are the genuinely untested extensions this program should investigate (Section 9's three recommended scopes, below, are unchanged by this clarification).

Feasibility of each option, given the verified config mechanism:

1. **Qwen decoder LoRA** — ALREADY THE CURRENT BASELINE (q/v only). Extending `target_modules` to `["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]` is a one-line YAML change — **feasible, cheap.**
2. **Encoder LoRA** — the encoder (`^perception\\.encoder\\..+$`) is currently in `freeze_params`; removing it from that list and adding a LoRA block targeting the FastConformer's attention projections would need the encoder to be a PEFT-wrappable module — **needs verification against `nemo.collections.speechlm2.modules.perception`'s actual class structure before assuming it's a simple config change.**
3. **Encoder + decoder LoRA combined** — combination of 1+2, same feasibility caveat as 2.
4. **Full decoder fine-tuning** — remove `^llm\\..+$` from `freeze_params` and drop the `lora:` block entirely. Qwen3-1.7B has ~1.7B params; **fine-tuning all of them in fp16/bf16 on 4×11GB GPUs via FSDP is a real VRAM risk that must be checked with a 1-step smoke test before committing GPU-hours to a full run** — this is exactly the kind of assumption the instructions told you not to make.
5. **Full model fine-tuning (encoder+decoder+everything)** — almost certainly infeasible on this hardware (2.87B total params, full fp16 optimizer states); **not recommended without a feasibility spike, and likely DROP given the 11GB constraint.**
6. **Frozen encoder + decoder LoRA (current baseline)** — same as #1, already running.
7. **Baseline current configuration** — the reference point for all comparisons; needs no new work.

### Recommended 1–3 scopes to actually run (per your own instruction not to test every combination):
- **(a) Current baseline** (decoder LoRA, q/v only) — reference point, no new training needed if last session's UWB-ATCC number is trusted, or a fresh confirmatory run if you want a matched-seed comparison set.
- **(b) Broader decoder LoRA** (q/v/k/o + MLP projections) — cheapest new experiment, directly tests "is q/v-only LoRA under-parameterized for this domain shift."
- **(c) Full decoder fine-tuning** (drop LoRA, unfreeze `llm`) — **gated on a VRAM feasibility smoke test first** — this is the experiment that most directly tests the paper's "frozen decoder is the bottleneck" hypothesis, so it's high-value, but only if hardware allows it.

Encoder unfreezing (options 2/3) is **deprioritized**: even setting aside the current config's invalidity, unfreezing a FastConformer encoder pretrained on 234k hours of general audio risks catastrophic forgetting on a ~10h fine-tuning set, and doesn't test the paper's actual stated hypothesis (which is about the *decoder*, not the encoder). Recommend DROP unless (a)/(b)/(c) leave the question genuinely open.

---

## 10. Wav2Vec2 Regularization Study

**[VERIFIED]** exact confound, from diffing `salm_uwb_atcc.yaml` → `salm_uwb_atcc_v3.yaml`:
1. `lora_dropout`: 0.01 → 0.1
2. SpecAugment: none → freq_masks=2, time_masks=10
3. `weight_decay`: 1e-3 → 1e-2

(Note: this is the **Canary-Qwen** config, not W2V2 — the review's "regularization ablation" concern applies to the Canary-Qwen v1→v3 transition, not W2V2, which has no analogous multi-variable regularization change in this repo. Section 8's title in the original request says "Wav2Vec2 Regularization Study" but the actual confound the reviewers describe is in Canary-Qwen's v1→v3 transition — flagging this mapping explicitly so the redesigned paper doesn't misattribute the ablation to the wrong model.)

**Minimum sequential ablation** (not a full 2³ factorial — 8 runs is wasteful when the components are likely to have very different effect sizes):

1. **Run 0 (reference):** v1-equivalent — `lora_dropout=0.01`, no SpecAugment, `weight_decay=1e-3`. (Already have this: 23.32%.)
2. **Run 1: SpecAugment only** (dropout=0.01, wd=1e-3, +SpecAugment). SpecAugment is the most likely single largest contributor on a 10-hour dataset (acts directly on the audio input, addresses limited data variety) — test it first.
3. **Run 2: dropout only** (dropout=0.1, wd=1e-3, no SpecAugment).
4. **Decision gate:** if Run 1 alone recovers most of the v1→v3 gap (23.32%→~21%), weight_decay's marginal contribution is likely small — a Run 3 (weight_decay only) becomes optional rather than required, saving one GPU-hour slot. If Run 1 and Run 2 each explain only a small fraction, run Run 3 (weight_decay=1e-2 only) to complete the picture before concluding anything about interaction effects.
5. **Optional Run 4:** best single-factor + one additional factor (not all three), only if Runs 1–3 don't already explain the v1→v3 gap.

This sequential design extracts the causal decomposition reviewers asked for in 2–3 runs typically, not 8, by using early results to decide whether further runs are informative — directly implementing your own "avoid experiments that only create more tables" instruction.

---

## 11. Speaker-Independent Study

**[VERIFIED, this session — EXP-007]** Speaker sets: `train_female`={gf1,zf1,zf2}, `test_female`={zf3}; `train_male`={sm1-4}, `test_male`={gm1,gm2}. Recording/utterance overlap = 0 in both directions (verified via `comm -12` on segments/text). No transcript leakage (0 shared utterance IDs). **Language-model leakage risk exists and is unmitigated**: the only trained ATCOSIM KenLM was built on the leaked full-corpus split (`ISS-005`) — it must not be used for LM-fused decoding of `test_female`/`test_male` without retraining on `train_female`/`train_male` text only.

**Small held-out speaker count is a real limitation the paper must state, not hide:** `test_female` = **1 speaker** (zf3, 616 utterances), `test_male` = **2 speakers** (gm1, gm2, 640 utterances combined). A single-speaker test set means the female WER (4.85%) is really "WER on one held-out speaker," not a population estimate — session-level or utterance-level bootstrap confidence intervals should be reported (Section 13) rather than a bare point estimate, and the paper should explicitly say "n=1 held-out speaker" for the female split rather than implying broader generalization. The male split (n=2) allows a per-speaker breakdown (gm1 vs. gm2 individually) that has NOT yet been computed — recommended before finalizing any claim, since if the two speakers differ substantially, the pooled 19.97% could be misleading (e.g., driven by one much-harder speaker).

**Recommended framing for the paper:** describe this explicitly as "speaker-disjoint evaluation on a small number of held-out speakers per gender (1 female, 2 male)," not "speaker-independent generalization" without qualification — the sample size doesn't support a strong generalization claim, only a directionally-informative one that corrects the previously-leaked numbers.

---

## 12. Cross-Corpus Study

UWB-ATCC (8kHz, real ATC, noisy, broad vocabulary, Czech-accented controllers) and ATCOSIM (32kHz→16kHz, simulated, clean close-talk, scripted phraseology, German/Swiss accents) differ on **sample rate, recording conditions, vocabulary breadth, and speaker population simultaneously** — a cross-corpus WER number would conflate all of these, making it hard to attribute a large WER jump to "domain shift" specifically versus "different microphone/sample-rate/vocabulary entirely."

- **Transcript conventions:** not verified to be compatible in this session — would need a direct text-normalization diff before any cross-corpus run; flagged as an open question, not assumed compatible.
- **Preprocessing match:** ATCOSIM is resampled 32kHz→16kHz already; UWB-ATCC is natively 8kHz and would need upsampling to 16kHz to match — introduces its own artifacts, another confound.
- **LM vocabulary leakage:** any KenLM trained on one corpus's transcript text would need to be either dropped entirely or clearly caveated when tested cross-corpus.
- **Does fine-tuning invalidate "unseen domain"?** Yes, definitionally — a model fine-tuned on ATCOSIM and tested on UWB-ATCC is testing cross-corpus transfer of a fine-tuned model, not zero-shot domain generalization; this is a real, answerable question but must be labeled as "cross-corpus transfer after in-domain fine-tuning," not "unseen domain performance."

**Verdict: methodologically weak as a headline claim, given the confounds above, and moderately expensive** (requires either a full second fine-tuning run per model per direction, or a zero-shot cross-eval which conflates domain shift with everything else). **Recommendation: treat as OPTIONAL/exploratory** — one direction only (UWB-ATCC→ATCOSIM, the more realistic transfer direction, since UWB-ATCC is the larger/harder/more general corpus) as a single supplementary result if compute allows after the P0/P1 experiments, explicitly labeled "cross-corpus transfer," never "unseen domain."

---

## 13. Hardware/Software Optimization

**[VERIFIED, this session]**
- CPU: 24 cores. RAM: 125GB (121GB available). Disk: 1.3TB free of 1.8TB.
- GPUs: 4× RTX 2080 Ti, 11GB each, driver 610.43.02.
- `w2v2_asr` env: torch 1.13.0+cu117, transformers 4.24.0, datasets 2.14.0.
- `canary_ft` env: torch 2.6.0+cu124, transformers 4.51.0, peft 0.14.0, nemo 2.8.0rc0.

**Assessment (extending `AUD-004` from this session's earlier repo audit):**
- **Efficient:** current W2V2 large-model configs (batch=1 at the VRAM floor, DDP correctly chosen over DataParallel) and Canary-Qwen FSDP configs (correctly chosen over DDP, which would OOM a 2.87B model).
- **Overkill (confirmed, `AUD-004`):** the UWB-ATCC W2V2-base 3000-step "pipeline validation only" run used all 4 GPUs when 1 would have validated the same thing.
- **Underkill in process (confirmed, `AUD-004`):** the Canary-Qwen "Lower LR" (1e-4, full 10k steps) ablation ran to a fixed, non-adaptive step count despite producing a clearly bad trajectory (32.58% WER) that was likely visible well before the final step. (The separate "research-optimized" 3e-5 ablation is dropped from the active record as of 2026-09-08 — user decision — and no longer part of this comparison.)
- **New for this program:** every experiment in Section 16's roadmap should use a **staged step-count protocol** (short checkpoint → decision gate → extend or stop), not a fixed full-length run, given the historical evidence that this would have saved GPU-hours on at least two prior ablations without losing information.
- **Not recommended:** switching frameworks, upgrading to a newer transformers/PyTorch stack mid-program, or otherwise changing infrastructure "because it's cleaner" — the current versions are pinned for specific, documented compatibility reasons (`ENVIRONMENT.md` ENV-002/ENV-003) and changing them introduces new risk with no research benefit.
- **Bottleneck not yet characterized:** actual dataloader/CPU-vs-GPU utilization during a live run was not measured in this session (would require a live smoke test, which was out of scope for a planning-only task) — recommend a short `nvidia-smi dmon` + `htop` capture during Stage 1 smoke tests (Section 14) to confirm GPUs aren't idling on data loading, rather than assuming either way.

---

## 14. Statistical Analysis Plan

Given the small held-out-speaker counts (Section 11) and the historically confounded ablations (Section 10), the analysis needs to do more than report point-estimate WER:

- **Per-speaker WER** for all speaker-independent results (mandatory given n=1/n=2 held-out speakers).
- **Bootstrap confidence intervals** (utterance-level resampling, e.g. 1,000 resamples) for every headline WER number — cheap to compute post-hoc from existing hypothesis/reference pairs, no retraining needed.
- **Paired bootstrap comparison** between any two systems evaluated on the *same* test set (e.g., W2V2 vs. Canary-Qwen on UWB-ATCC test) to state whether an observed WER gap is statistically distinguishable from noise, not just numerically different.
- **Single-seed runs are acceptable for the adaptation-scope study (Section 9)** given the compute budget — reserve repeated seeds only for the single most decision-critical comparison (Section 6's chosen contribution), not across the board, per your own "repeated seeds only when scientifically justified" instruction.
- **Learning curves as figures**, not tables (directly resolves Reviewer 4Pd7's presentation concern) — plot WER-vs-step for the adaptation-scope and regularization ablations.
- **Compute-vs-performance**: report GPU-hours-to-target-WER alongside final WER for the adaptation-scope comparison (ties into Candidate D from Section 5 as a secondary framing).
- **Do NOT** add a full statistical battery (e.g., ANOVA across every factor) merely for appearance — the sequential, gated design in Sections 9/10 already isolates single-factor effects; bootstrap CIs and paired comparisons are the simplest defensible tools for the actual claims being made.

---

## 15. Manuscript Redesign

- **Title:** remove "Unseen ... Domains." **Reconciled against the manuscript's actual usage:** the term is used in two different, both internally-defensible senses — narrowly and correctly in Future Work ("unseen ATC corpora" = cross-corpus, Sec. VII), and more broadly in the title, plausibly meaning "unseen relative to general-purpose pretraining" (true — LibriSpeech/general web text never saw ATC audio). The defect isn't that either reading is false; it's that the title's broad reading is what three of four reviewers understood, and the 80/20 same-corpus split doesn't support *that* reading. Recommended: *"Adaptation Scope and Regularization in a Hybrid Speech-LLM for Air Traffic Control Speech Recognition: A Controlled Study"* (or similar) — centers the actual, defensible contribution (Section 6) and removes the ambiguous term entirely rather than trying to clarify it in place.
- **Abstract/Research question:** reframe around Candidate B (adaptation scope), with speaker-independent evaluation and the apples-to-apples decoding fix as supporting pillars — drop the "which architecture is better" framing entirely.
- **Contributions:** (1) a controlled decoder-adaptation-scope ablation on Canary-Qwen for ATC speech, isolating LoRA-scope and full-decoder-FT effects; (2) a corrected, decoding-fair comparison protocol against a fully fine-tuned CTC baseline; (3) a properly component-ablated regularization analysis; (4) a corrected, small-n-disclosed speaker-independent ATCOSIM evaluation.
- **Related work:** should now engage with ILME/density-ratio LM fusion and N-best rescoring literature (Section 8), and with existing decoder-adaptation-scope studies for hybrid SALMs if any exist in the literature — a literature check for this is recommended but out of scope for this planning report.
- **Methodology:** state the freeze/LoRA configuration precisely (Section 9's verified mechanism), state the exact decoding-fairness protocol chosen (Section 8), and state the speaker-independent split's small-n limitation explicitly (Section 11) rather than as a footnote.
- **Results:** headline table should be the adaptation-scope comparison (3 configs × 1–2 corpora), not the original architecture-vs-architecture table. The original architecture comparison can remain as a *secondary, clearly-labeled* reference point ("for context, a fully fine-tuned CTC baseline achieves X%") rather than the primary claim.
- **Discussion:** connect adaptation-scope results back to the "frozen decoder bottleneck" hypothesis with actual evidence (Section 9), rather than asserting it. State explicitly what the decoding-fairness fix did or didn't change about the original comparison's conclusion.
- **Limitations:** speaker-independent small-n (Section 11), cross-corpus not attempted or attempted only as exploratory (Section 12), single-seed runs outside the primary comparison (Section 14), ATCOSIM Canary-Qwen config gap now closed by pushing it to GitHub (a concrete, cheap fix — recommend doing this regardless of the rest of the program).
- **Conclusion:** should state what was learned about adaptation scope specifically, not re-assert the original "W2V2 beats Canary-Qwen" framing as a general architectural claim.
- **Figures:** replace Table II (W2V2 step-by-step) and Table III (Canary WER-vs-step) with line plots (Section 14) — training diagnostics move to an appendix or a compact supporting figure, not a primary results table (resolves Reviewer 4Pd7's Concern #10/#11 directly).
- **Mechanical fixes** (do regardless of the rest): fix section ordering (Related Work after Introduction), fix the PDF title metadata placeholder, **label the 13h and 10.5h figures explicitly as train+test vs. train-only respectively (both are correct once labeled — Section 2, item 12)**, fix cited typos and the mangled reference, correct "six training runs" (Sec. IV.C) to the actual count (7, per Section 3.G).
- **Acknowledgements/AI-disclosure update required, not optional:** the current manuscript states *"The AI system did not execute any experiments or access the training hardware directly."* **[VERIFIED FACT]** — that was true for the original paper's runs, but is **not** true of EXP-007 (this session), where Claude directly launched, monitored, and verified the speaker-independent ATCOSIM training runs at the authors' explicit instruction. If EXP-007's results are used in the redesigned paper, the disclosure text must be updated to accurately describe this — the venue's AI-use disclosure policy is a stated commitment in the current draft, and citing EXP-007 without correcting it would make the disclosure inaccurate.

---

## 16. Final Experiment Roadmap

Staged per your Part 12 structure. GPU-hour estimates use this session's verified baseline rate (~0.365 steps/sec for W2V2-large DDP on this hardware, extrapolated proportionally for Canary-Qwen using the existing documented ~5.3h/10k-step rate).

### STAGE 0 — Dataset & environment validation
- **EXP-ID:** S0-1 — QUESTION: Are transcript conventions/preprocessing compatible enough for a future cross-corpus attempt? — Not required unless Stage 6 is greenlit. DEFER.
- **EXP-ID:** S0-2 — QUESTION: Does `SALM.generate()` support N-best decoding? — **P0, required before Section 8's decoding-fairness experiment can be scheduled.** No GPU-hours (code inspection only). SUCCESS: N-best supported → proceed with rescoring design. FAILURE: not supported → fall back to native-vs-native comparison (Section 8 fallback).
- **EXP-ID:** S0-3 — QUESTION: Does full decoder fine-tuning fit in 11GB×4 via FSDP? — **P0, required before Section 9(c) can be scheduled.** 1-step smoke test, ~5 min, <0.01 GPU-hours. FAILURE: OOM → drop option (c), keep (a)/(b) only.

### STAGE 1 — Cheap smoke tests
- **EXP-ID:** S1-1 — all Stage 2+ configs, 10–20 steps each, 1 GPU where the config allows (per your compute-minimization policy) — confirms config/data/checkpoint I/O before committing full runs. P0.

### STAGE 2 — Baseline reproduction (confirmatory, not exploratory)
- **EXP-ID:** S2-1 — W2V2-large UWB-ATCC, full config, 1 seed — confirms the 14.54%/12.69% anchor still reproduces on current environment. P1 (skip if you trust last session's provenance-verified number and want to save ~8.6h).
- **EXP-ID:** S2-2 — Canary-Qwen UWB-ATCC baseline (current LoRA q/v-only config) — confirms the 23.32% anchor. P1, same logic as S2-1.

### STAGE 3 — High-information ablations
- **EXP-ID:** S3-1 — QUESTION: does SpecAugment alone recover most of the v1→v3 gap? — HYPOTHESIS: yes, largest single factor. DATA: UWB-ATCC. MODEL: Canary-Qwen. ADAPTATION: baseline LoRA q/v. DECODING: native. CONFIG: dropout=0.01, wd=1e-3, +SpecAugment. SEED: 1234. GPU COUNT: 4 (FSDP required, per DEC-004). EXPECTED WALL-CLOCK: ~5.3h (matches v1/v3 historical rate). EXPECTED GPU-HOURS: ~21. DEPENDENCIES: none. SUCCESS: WER materially closer to v3's 20.70% than to v1's 23.32%. STOP: if WER ≈ v1, SpecAugment isn't the driver — proceed to S3-2 regardless. SCIENTIFIC VALUE: isolates the largest suspected regularization factor. REVIEWER CONCERN: #3 (confounded regularization). **P0.**
- **EXP-ID:** S3-2 — dropout-only variant, same structure as S3-1 but dropout=0.1 alone. **P0** (run in parallel-scheduled sequence with S3-1, not simultaneously on the same 4 GPUs).
- **EXP-ID:** S3-3 — weight_decay-only variant. **P1, conditional**: only run if S3-1+S3-2 together don't explain most of the v1→v3 gap (decision gate, Section 10).
- **EXP-ID:** S3-4 — broader decoder LoRA (q/v/k/o/MLP), baseline regularization. HYPOTHESIS: q/v-only LoRA is under-parameterized for domain shift. **P0** — directly answers Reviewer 4Pd7 Concern #2.
- **EXP-ID:** S3-5 — full decoder fine-tune (gated on S0-3 passing). HYPOTHESIS: unfreezing the full decoder closes most of the remaining gap to W2V2. **P0 if S0-3 passes, else DROP.**

### STAGE 4 — Fair model comparison
- **EXP-ID:** S4-1 — W2V2 greedy vs. Canary-Qwen native (no external LM either side) — the honest baseline comparison. **P0.**
- **EXP-ID:** S4-2 — Canary-Qwen + N-best rescoring with in-domain KenLM (gated on S0-2 passing). **P1 if feasible, else DROP with justification in the paper (Section 8 fallback).**

### STAGE 5 — Speaker-independent evaluation
- **Already complete (EXP-007, this session).** Remaining work: per-speaker (gm1 vs gm2) breakdown, bootstrap CIs (Section 14) — **P0, no new training, analysis only.**

### STAGE 6 — Cross-corpus (conditional)
- **EXP-ID:** S6-1 — UWB-ATCC-fine-tuned model evaluated on ATCOSIM test — **P2, optional**, only after Stages 3–5 are complete and if GPU-hour budget remains. Explicitly label as "cross-corpus transfer," not "unseen domain," per Section 12.

### STAGE 7 — Final confirmatory runs
- Re-run whichever Stage 3 configuration is chosen as the paper's headline adaptation-scope result with a second seed, ONLY for that one configuration (per Section 14's single-seed-except-headline policy). **P1.**

### Adaptive decision tree
```
S0-2 (N-best supported?) --YES--> schedule S4-2
                          --NO--> DROP S4-2, use native-vs-native only (S4-1) as the primary decoding comparison

S0-3 (full-decoder-FT fits in VRAM?) --YES--> schedule S3-5
                                     --NO--> DROP S3-5, adaptation-scope study becomes (a) baseline vs (b) broader-LoRA only

S3-1 + S3-2 (jointly explain >80% of v1→v3 gap?) --YES--> DROP S3-3 (weight_decay), saves ~21 GPU-hours
                                                 --NO--> run S3-3 to complete the decomposition
```

### Estimated total compute for the P0-only program
S1 (smoke tests, ~10 configs × ~15 min × 1 GPU) ≈ 2.5 GPU-hours. S3-1/S3-2/S3-4 (3 × ~21 GPU-hours) ≈ 63 GPU-hours. S3-5 (conditional, ~21 GPU-hours if VRAM allows). S4-1 (analysis only on existing checkpoints, ~0 new GPU-hours if S2 skipped). Stage 5 (analysis only, ~0). **P0 total: ~65–86 GPU-hours** depending on the S0-3 gate outcome — roughly 4–5.5 wall-clock days at 4-GPU utilization, or faster if runs are scheduled back-to-back without idle time between stages.

---

## 17. Compute Budget Summary

| Stage | GPU-hours (P0 only) | Wall-clock (4 GPUs) |
|---|---|---|
| Stage 0 (spikes) | ~0.05 | ~15 min |
| Stage 1 (smoke tests) | ~2.5 | ~40 min |
| Stage 3 (ablations, P0 subset) | ~63–84 | ~16–21h |
| Stage 4 (fair comparison) | ~0 (reuses Stage 3/existing checkpoints) | — |
| Stage 5 (speaker-independent) | ~0 (analysis only) | — |
| Stage 7 (confirmatory re-seed) | ~21 (one config only) | ~5.25h |
| **Total (P0)** | **~86–108 GPU-hours** | **~5–6 wall-clock days** |

P1/P2/optional items (S2 reproduction, S6 cross-corpus) would add roughly another 30–50 GPU-hours if all pursued — not recommended to schedule until P0 results are in hand and reviewed.

---

## 18. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Full decoder fine-tune OOMs on 11GB×4 | Medium | Blocks S3-5, weakens the adaptation-scope story's strongest data point | S0-3 smoke test before committing; fallback to (a)/(b) only comparison |
| N-best generation unsupported by installed NeMo | Medium | Blocks the cleanest decoding-fairness fix | S0-2 spike; fallback to native-vs-native honest comparison (still valid, just less complete) |
| Broader decoder LoRA (S3-4) doesn't meaningfully change WER | Medium | Weakens the "adaptation scope matters" narrative | Still scientifically valid as a negative result — report it as such, don't suppress it |
| Speaker-independent per-speaker breakdown reveals gm1/gm2 are wildly different | Low-medium | Complicates the male-split headline number | Already flagged in `EXPERIMENTS.md`; report per-speaker numbers transparently rather than only pooled |
| ATCOSIM Canary-Qwen config, once pushed to GitHub, differs subtly from what actually ran historically | Low-medium | Could mean the 7.06%/3.33% numbers aren't exactly reproducible either | Diff the `w2v2-air-traffic` working config against any historical run logs/checkpoints before citing as reproducible; treat with the same skepticism applied to the encoder-unfrozen config in this report |
| Compute budget (86-108 GPU-hours) exceeds available time before a resubmission deadline | Medium | Forces scope cuts | Stage-gate structure (Section 16) already designed to drop the most expensive, least-certain item (S3-5) first if time runs short |

---

## 19. Criteria for Stopping Experiments

- Stop a Stage-3 regularization component run early if training loss diverges or eval WER is monotonically worsening past step ~1000 (matches this repo's own historical evidence that bad configs separate early — Section 13).
- Stop the adaptation-scope study at 2 configs (a)+(b) if S0-3 fails and S3-4 alone already shows a clear, interpretable trend — don't force a third data point that requires infeasible hardware.
- Do not proceed to Stage 6 (cross-corpus) if Stage 3 compute already exceeds ~100 GPU-hours and a submission deadline is approaching — Section 12 already establishes this as optional/exploratory, not required for a defensible paper.
- Treat the encoder-unfrozen ablation as permanently dropped, not "to be re-run," unless a specific reviewer or the authors decide the encoder-unfreezing question is worth the additional ~21 GPU-hours on top of the P0 program — it is not part of the P0 recommendation.

---

## 20. Model / Agent / Skill Usage

MODEL: Sonnet 5 throughout this planning task.
AGENTS: None — this entire investigation (repo/upstream code reading, hardware/software audit, config diffing, duration computation) was done directly with `Bash`/`Read` in this session, not delegated, because the task required cross-referencing context already built earlier in this conversation (repo relationships, EXP-007 results, prior audit findings) that a fresh agent would not have had.
SKILLS: None explicitly invoked this task (no `engineering-memory` skill call was needed — historical records were retrieved directly via `grep`/`Read`, which is the cheaper path the skill itself recommends when deterministic retrieval suffices).
TOOLS: `Bash` (nvidia-smi, free, df, nproc, conda run, md5sum, diff, python duration computation), `Read`, `Write`.

## 21. Token/Resource Usage

Exact task-level token usage unavailable from exposed telemetry.

---

## Executive Recommendation

1. **What the paper should now claim:** a controlled adaptation-scope study of Canary-Qwen (LoRA q/v-only → broader LoRA → full decoder fine-tune, if feasible) for ATC speech, paired with a decoding-fairness-corrected comparison against a fully fine-tuned W2V2-large CTC baseline and a properly component-ablated regularization analysis — NOT a general "which architecture is better" claim, and NOT an "unseen domain" claim.

2. **Experiments absolutely required (P0):** Stage 0 spikes (N-best support, full-decoder-FT VRAM feasibility — free, code-only), Stage 3 sequential regularization ablation (SpecAugment-only, dropout-only, conditionally weight-decay-only), Stage 3 broader-decoder-LoRA run, Stage 4 honest native-vs-native decoding comparison, Stage 5 speaker-independent per-speaker analysis (already have the data, just needs the breakdown).

3. **Experiments to avoid:** the full 2³ regularization factorial, encoder-unfreezing re-runs (unless independently prioritized), a naive shallow-fusion of the KenLM onto Canary-Qwen, symmetric bidirectional cross-corpus evaluation, comparison against a third architecture (Whisper) — all correctly out of scope for turning this into a defensible paper within reasonable compute.

4. **Estimated total GPU-hours:** ~86–108 for the P0 program (~5–6 wall-clock days on this 4-GPU machine), before any P1/optional additions.

5. **Expected timeline/order:** Stage 0 (same day, no GPU cost) → Stage 1 smoke tests (same day) → Stage 3 ablations (3–4 days, largest compute block) → Stage 4/5 analysis (same day, no new training) → Stage 7 single re-seed confirmatory run (~1 day) → manuscript redesign and figures (parallel, no GPU cost).

6. **Biggest remaining scientific risk:** that even a fully-fine-tuned Canary-Qwen decoder doesn't close the gap to W2V2, in which case the paper's contribution shifts from "adaptation scope explains the gap" to "adaptation scope alone does not explain the gap, implicating architecture/pretraining more fundamentally" — this is still a publishable, honest finding, but it's a materially different narrative than the current draft assumes, and the manuscript redesign (Section 15) should be written to accommodate either outcome rather than presupposing one.

7. **Single most important experiment:** **S3-5 / S0-3 pairing** — the full-decoder-fine-tune feasibility spike and (if it passes) the full-decoder-fine-tune run itself. This is the one experiment that directly tests the paper's own central, currently-unsupported interpretive claim (frozen decoder = bottleneck), and its outcome determines which of the two narratives in point 6 the paper ends up telling.
