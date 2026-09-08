# Final Research Program — Execution-Ready Adaptive Plan

Status: PLANNING ONLY. No training launched. No source code, scripts, datasets, checkpoints, or existing results modified. Nothing committed or pushed.

This document supersedes `IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` as the master execution plan. That document is preserved unchanged (per engineering-memory policy — never delete/silently rewrite) and remains the fuller narrative discussion of reviewer-concern classification and manuscript reconciliation. This document restructures its conclusions into a dependency graph, adaptive decision tree, and stage-gated execution plan, and adds one materially new, load-bearing finding from direct code inspection this session (below).

**New this session:** `nemo/collections/speechlm2/models/salm.py:408` — `SALM.generate()` passes `**generation_kwargs, generation_config=generation_config` directly to `self.llm.generate()` (a standard HuggingFace `generate()` call on the Qwen3-1.7B decoder). The docstring itself demonstrates `GenerationConfig(do_sample=True, num_beams=5)` as supported usage. **N-best generation is CONFIRMED FEASIBLE with zero code changes** — this was the single largest unresolved technical uncertainty in the prior report, and it resolves favorably. This is a **VERIFIED FACT** (read directly from installed source, not inferred), and it materially changes the decision tree in Section 4/5 below.

---

## 1. Reconcile the Current Research Plan

**Already proven (VERIFIED FACT, no further work needed):**
- The v3 Canary-Qwen config changes 3 regularization variables simultaneously (`lora_dropout`, SpecAugment, `weight_decay`) — confirmed by diff.
- The current Canary-Qwen baseline already does decoder-only LoRA on `q_proj`/`v_proj` — confirmed by config + PEFT semantics + manuscript Sec. III.C agreement.
- N-best generation is supported natively — confirmed by source code, this session.
- Speaker-independent ATCOSIM splits are leakage-free (0 recording/utterance overlap, verified via `comm`) — `EXP-007`, `VAL-006`/`VAL-007`.
- UWB-ATCC exact durations: train=10.54h, test=2.63h — computed directly from `segments` files, resolves the manuscript's internal 13h/10.5h figures (both correct, unlabeled subset).

**Remains uncertain (genuine open questions, not yet resolvable without either more code-reading or a smoke test):**
- Whether full Qwen3-1.7B decoder fine-tuning fits in 11GB×4 via the existing FSDP config (back-of-envelope: LoRA trains 27.8M params; full fine-tune trains ~1.7B — roughly 60× more optimizer-state memory for the trainable set. FSDP shards this across 4 GPUs, but whether the per-shard remainder plus the frozen encoder plus activations fits under 11GB cannot be determined by arithmetic alone — genuinely needs an empirical spike).
- Whether the encoder-unfrozen historical result (23.82% WER, 838.8M params) reflects a real run under a since-lost config, or something else — checkpoints are gone, so this is permanently uncertain absent author recollection (`ISS-007`).
- Whether ATCOSIM and UWB-ATCC's transcript/preprocessing conventions are compatible enough for a defensible cross-corpus number — not yet checked at the text-normalization level.

**Dependencies:**
- The decoding-fairness experiment (Section 5) depends on nothing else — it can run against already-existing checkpoints if any survive, or against freshly-trained baselines from Section 6.
- The adaptation-scope study (Section 6) depends on the full-decoder-FT VRAM spike only for its most expensive branch (full fine-tune); the LoRA-scope branches have no dependency.
- The regularization study (Section 7) depends on nothing else and can run in parallel with Section 6's LoRA-scope branches (both use the same 4-GPU FSDP setup, so they are queue-parallel, not GPU-parallel, on this single 4-GPU machine).
- Cross-corpus evaluation (Section 10) depends on the adaptation-scope study producing a "final" model configuration worth testing cross-corpus — running it earlier would mean re-running it later.
- Manuscript reconstruction (Stage 6) depends on all experimental stages completing.

**Redundant (do not re-run):**
- UWB-ATCC W2V2-large baseline reproduction — already have a provenance-verified number (14.54%/12.69%); re-running only buys a fresh-seed confirmatory data point, not new information. Demoted to P2/optional.
- ATCOSIM W2V2-large baseline — same logic, P2/optional.
- Canary-Qwen UWB-ATCC LoRA-only baseline (23.32%) — already have this number with verified provenance (config present, matches manuscript). No need to re-run unless it's needed as the literal control arm for a new ablation using an identical random seed — see Section 7.

**Answerable without training (analysis-only, zero GPU-hours):**
- Per-speaker WER breakdown for `test_male` (gm1 vs. gm2) from EXP-007's existing predictions.
- Bootstrap confidence intervals on all four EXP-007/historical WER numbers, from existing hypothesis/reference pairs.
- The decoding-fairness method selection (Section 5) — resolved this session by code inspection alone.
- The cross-corpus feasibility verdict (Section 10) — resolved by inspecting transcript/preprocessing pipelines, not by training.

**Answerable with a small feasibility test (near-zero GPU cost, <30 min, often <1 GPU):**
- Full-decoder-FT VRAM fit (Spike D).
- N-best decode + KenLM rescore mechanics on a handful of existing/held checkpoint(s) or a 10-step smoke checkpoint (Spike A/B combined).
- Broader-target LoRA config sanity check (does PEFT accept `target_modules` including MLP projections on this Qwen3 architecture) — Spike C.

**Requires full training (the actual GPU-hour budget):**
- Regularization component ablation (Section 7).
- Adaptation-scope comparison runs themselves (Section 6), once gated.
- Any confirmatory re-seed run (Stage 5).

### Dependency graph

```
                         ┌─────────────────────────┐
                         │  Stage 0: repo/env audit │  (DONE, this + prior session)
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┼────────────────────────┐
                 ▼                    ▼                        ▼
        Spike A/B (N-best/LM)  Spike C (broader LoRA    Spike D (full-decoder-FT
        CONFIRMED FEASIBLE      config sanity)           VRAM fit) — NOT YET RUN
        (code-verified)         near-zero cost                near-zero cost
                 │                    │                        │
                 ▼                    ▼                        ▼
      Decoding-fairness      Adaptation-scope Branch 2   Adaptation-scope Branch 3
      experiment (Sec.5)     (broader LoRA) — P0         (full decoder FT) — P0 if
      — P0, no further gate                              Spike D passes, else DROP
                 │                    │                        │
                 └────────┬───────────┴────────────┬───────────┘
                          ▼                         ▼
              Regularization ablation      Speaker-independent analysis
              (Sec. 7) — P0, independent   (Sec. 9) — DONE (EXP-007),
              of adaptation-scope branch    needs per-speaker breakdown only
                          │                         │
                          └────────────┬────────────┘
                                       ▼
                         Cross-corpus evaluation (Sec. 10)
                         — P2/conditional, only after above
                                       │
                                       ▼
                         Stage 5: confirmatory re-seed run
                         (chosen headline config only)
                                       │
                                       ▼
                         Stage 6: manuscript reconstruction
```

---

## 2. The Scientific Core

**Chosen question:** *"Within a fixed hybrid Speech-LLM architecture (Canary-Qwen-2.5B), how does decoder adaptation scope (attention-only LoRA → broader LoRA → full decoder fine-tuning, if feasible) affect ATC speech recognition accuracy, and under a decoding-fairness-corrected comparison, how much of the residual gap to a fully fine-tuned CTC baseline (Wav2Vec2-large) does adaptation scope explain?"*

This is NOT "which model is better" — it explicitly frames Canary-Qwen's own internal adaptation-scope sensitivity as the primary object of study, with the W2V2 comparison serving as a fixed external reference point, not a symmetric competitor claim.

**What this study CAN isolate:**
- Adaptation scope (within Canary-Qwen, holding architecture/scale/pretraining/data fixed) — this is a genuinely controlled, single-factor comparison.
- Regularization components (within Canary-Qwen's v1→v3 transition, via the sequential ablation in Section 7) — controlled, single-factor-at-a-time.
- Decoding fairness (native vs. N-best+KenLM, both models) — controlled to the extent both models are evaluated under a comparable no-external-LM and matched-external-LM condition.
- Speaker generalization (ATCOSIM gender-disjoint splits) — controlled, verified leakage-free.

**What this study CANNOT isolate, and must say so explicitly:**
- Architecture vs. adaptation-scope as separate causes of the W2V2/Canary-Qwen gap — even a full-decoder-fine-tune Canary-Qwen run remains a different architecture (SALM vs. CTC), different scale, different pretraining corpus, from W2V2. Closing (or not closing) the gap under full fine-tuning narrows the *adaptation-scope* explanation but does not isolate architecture from pretraining-data-domain from optimization dynamics.
- General "unseen domain" generalization — the speaker-independent split (Section 9) supports a *speaker*-generalization claim; nothing in the current design supports a *domain* (i.e., different corpus/channel/vocabulary) generalization claim unless Section 10's cross-corpus evaluation is both feasible and run — see that section's verdict before assuming this claim is available.
- Cross-architecture data efficiency in general (Candidate A from the prior report) — deliberately out of scope for the chosen primary question; may be worth a narrow follow-up but is not part of this program (see Section 8's verdict).

---

## 3. Feasibility Gates First

### Spike A — Canary-Qwen N-best generation/extraction
**OBJECTIVE:** Confirm N-best hypotheses can be extracted from `SALM.generate()` without model modification.
**FILES/CODE TO INSPECT:** `nemo/collections/speechlm2/models/salm.py` (already done, this session — see header finding).
**COMMAND OR MINIMAL TEST:** Not required as a separate spike — **already resolved by code inspection**. If empirical confirmation is still wanted before committing compute: `model.generate(prompts=[...], generation_config=GenerationConfig(num_beams=5, num_return_sequences=5, max_new_tokens=128))` on a single existing test utterance, checking the returned tensor's batch dimension is 5× the input batch size (standard HF beam-search-with-return-sequences behavior).
**EXPECTED EVIDENCE:** Output tensor shape `(5, seq_len)` for a single input prompt; 5 distinct (or at least non-identical) decoded strings.
**GO CONDITION:** Output shape and content match expectation → proceed with N-best+KenLM rescoring design (Section 5).
**NO-GO CONDITION:** An exception is raised, or all 5 returned sequences are identical (indicating beam collapse or a config-passing bug) → fall back to native-vs-native comparison only (Section 5's fallback).
**FOLLOW-UP EXPERIMENTS ELIMINATED BY FAILURE:** The N-best+KenLM rescoring arm of Section 5; would not affect the adaptation-scope or regularization studies at all.
**STATUS: Effectively GO already, pending only a cheap empirical confirmation before scheduling real compute on top of it.**

### Spike B — Canary-Qwen external-LM-compatible decoding/rescoring
**OBJECTIVE:** Confirm that N-best hypotheses (from Spike A) can be rescored with the existing in-domain KenLM binary without any model-internal changes.
**FILES/CODE TO INSPECT:** `src/eval_model.py` (already understands KenLM scoring for W2V2, via `pyctcdecode`); confirm the KenLM binary format (`kenlm.Model` or `pyctcdecode`'s LM wrapper) can score arbitrary text strings independent of W2V2's CTC lattice — this is standard KenLM usage (`model.score(text)`), not W2V2-specific, so this is very likely a non-issue, but the exact scoring call needs to be written fresh for Canary-Qwen's plain-text hypotheses rather than reused verbatim from `eval_model.py`.
**COMMAND OR MINIMAL TEST:** `import kenlm; m = kenlm.Model(path); m.score("hypothesis text")` — a two-line sanity check, no GPU needed.
**EXPECTED EVIDENCE:** A finite log-probability score for a sample ATC-style sentence.
**GO CONDITION:** Scoring works → combine as `final_score = log p_SALM(y|x) [approximated by rank/generation score] + λ · log p_ext(y)`, select the top-scoring hypothesis per utterance.
**NO-GO CONDITION:** `kenlm` Python bindings aren't installed in `canary_ft` env, or the binary format is incompatible → install `kenlm` bindings (a pip install, not a code change) or fall back to native-only comparison.
**FOLLOW-UP EXPERIMENTS ELIMINATED BY FAILURE:** Same as Spike A's failure case.
**STATUS: Very likely GO — KenLM's Python API is corpus-agnostic; needs a 5-minute check, not a design question.**

### Spike C — Canary-Qwen broader decoder LoRA
**OBJECTIVE:** Confirm PEFT accepts an extended `target_modules` list (`q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj`) against the Qwen3-1.7B architecture used here, and that the resulting trainable-parameter count is sane.
**FILES/CODE TO INSPECT:** `salm_uwb_atcc.yaml`'s `lora:` block (already read, this session); PEFT's `LoraConfig` validation against the actual Qwen3 module names (needs one `python -c` check against the loaded model to confirm module names match, since Qwen3 naming could differ slightly from a generic Llama-style config).
**COMMAND OR MINIMAL TEST:** Load the Qwen3-1.7B model (already cached from prior runs) and `print([n for n,_ in model.named_modules() if 'proj' in n][:20])` to confirm the exact projection names, then dry-run `PeftModel.get_peft_model(model, LoraConfig(target_modules=[...]))` and print `model.print_trainable_parameters()`.
**EXPECTED EVIDENCE:** A trainable parameter count noticeably larger than 27.8M (extending from 2 projections to 7 should scale roughly linearly, i.e., ~3.5× → ~97M) with no exceptions raised.
**GO CONDITION:** Config loads cleanly, parameter count is in the expected range → proceed with Section 6 Branch 2.
**NO-GO CONDITION:** Module-name mismatch (unlikely, but architectures do vary) → adjust `target_modules` to the actual names found; this is a config fix, not a scope change, so failure here doesn't eliminate the experiment, only delays it by minutes.
**FOLLOW-UP EXPERIMENTS ELIMINATED BY FAILURE:** None — this spike essentially cannot hard-fail, only requires minor config correction.

### Spike D — Canary-Qwen full-decoder fine-tuning VRAM fit
**OBJECTIVE:** Determine empirically whether removing `^llm\..+$` from `freeze_params` and dropping the `lora:` block (i.e., training all ~1.7B Qwen3 decoder params) fits in 11GB×4 under the existing FSDP config (`tensor_parallel_size=1, data_parallel_size=4`).
**FILES/CODE TO INSPECT:** `salm_uwb_atcc.yaml`'s `freeze_params`/`lora` blocks (already read); FSDP sharding behavior is standard PyTorch, no custom code to inspect beyond the YAML.
**COMMAND OR MINIMAL TEST:** A **1-step** training run (`trainer.max_steps: 1` or equivalent override) with the modified config, watching for `CUDA out of memory` in the first optimizer step. This is the one spike that technically launches a training process — but at 1 step, it is a feasibility check, not an experiment, and should be treated as a Stage-1 smoke test (Section 15), not a P0/P1/P2 experiment in its own right.
**EXPECTED EVIDENCE:** Either a clean single optimizer step (VRAM fits) or an explicit `CUDA out of memory` error (doesn't fit).
**GO CONDITION:** 1 step completes without OOM → schedule Section 6 Branch 3 (full-decoder-FT) as a real multi-hour run.
**NO-GO CONDITION:** OOM on step 1 → **DROP** Branch 3 entirely; the adaptation-scope study becomes a 2-point comparison (baseline LoRA vs. broader LoRA) rather than 3-point, and the paper's contribution narrows accordingly (still valid, see Section 14's neutral/negative case framing).
**FOLLOW-UP EXPERIMENTS ELIMINATED BY FAILURE:** Section 6 Branch 3, Section 14's "full fine-tune closes the gap" narrative becomes unavailable — the paper must then rely on the LoRA-scope comparison alone to discuss the frozen-decoder hypothesis, with an explicit stated limitation that full fine-tuning could not be tested on this hardware.
**STATUS: Not yet run. This is the one gate genuinely requiring your explicit approval before I touch a GPU, since it does launch (an extremely short) training process.**

### Spike E — other critical implementation uncertainties
- **Whether the existing ATCOSIM lhotse cuts (`atcosim_train_cuts.jsonl`) already exist for a Canary-Qwen ATCOSIM speaker-independent run**, since EXP-007's speaker-independent work so far only covers W2V2 — if the paper wants Canary-Qwen evaluated under the same speaker-independent protocol (Section 9's open question), new lhotse cuts for `train_female`/`train_male`/`test_female`/`test_male` need to be built first. **[VERIFIED, this session]:** `~/w2v2-air-traffic/experiments/data/atcosim_corpus/lhotse/` only contains `atcosim_train_cuts.jsonl` and `atcosim_test_cuts.jsonl` (the full, leaked-split cuts) — **no gender-specific lhotse cuts exist yet.** This is a real, near-zero-cost prerequisite (data conversion, not training) that must happen before Canary-Qwen can be run on the speaker-independent split at all.

---

## 4. Adaptive Experiment Decision Tree

```
START
  │
  ├─ Spike A/B (N-best + KenLM rescoring): CONFIRMED FEASIBLE (code-verified, this session)
  │     └─ Proceed directly to Decoding-Fairness Experiment (Section 5) — no gate needed, run as part of Stage 2/4.
  │
  ├─ Spike C (broader LoRA config): near-certain GO, minutes to confirm
  │     └─ Proceed to Adaptation-Scope Branch 2 (broader LoRA) — P0, Stage 3.
  │
  └─ Spike D (full-decoder-FT VRAM, 1-step smoke test — REQUIRES APPROVAL, launches a process)
        │
        ├─ IF PASSES (no OOM at step 1):
        │     └─ Run Adaptation-Scope Branch 3 (full decoder FT, full run) — P0.
        │           ├─ IF full-FT WER ≈ closes gap to W2V2 (within ~2-3pp):
        │           │     └─ BEST-CASE narrative (Section 14) — adaptation scope is the dominant factor.
        │           │           Regularization ablation (Section 7) still runs regardless (independent question).
        │           │           Cross-corpus (Section 10) becomes higher-value as a generalization follow-up.
        │           └─ IF full-FT WER still trails W2V2 substantially (>5pp gap remains):
        │                 └─ NEUTRAL/NEGATIVE-CASE narrative (Section 14) — architecture/pretraining implicated
        │                       beyond adaptation scope. Regularization ablation still runs (still informative).
        │                       Cross-corpus becomes lower-value (a generalization claim on top of an
        │                       already-large architecture gap adds less) — consider dropping to P2.
        │
        └─ IF FAILS (OOM at step 1):
              └─ DROP Branch 3 entirely. Adaptation-scope study becomes 2-point (baseline vs. broader LoRA).
                    ├─ IF broader-LoRA WER improves substantially over baseline (>3pp):
                    │     └─ Still publishable: "adaptation scope matters even within LoRA's parameter budget,
                    │           though full fine-tuning could not be tested on this hardware" — a real, honest,
                    │           narrower contribution. GPU-hours saved: ~21 (no Branch 3 run).
                    └─ IF broader-LoRA WER is ≈ baseline (no meaningful change):
                          └─ Weakest outcome: adaptation-scope-within-LoRA doesn't explain much, and full-FT
                                couldn't be tested. Consider whether the regularization study (Section 7) alone
                                still supports a publishable contribution (it does — that ablation is independent
                                and answers its own reviewer concern regardless of this branch's outcome).

Regularization ablation (Section 7) — runs independently of the above, no gating dependency:
  Run 1 (SpecAugment-only) vs. Run 0 (v1-equivalent baseline, already have this number: 23.32%)
        │
        ├─ IF Run 1 alone recovers most of the v1→v3 gap (i.e., WER drops to ~21% or below):
        │     └─ Run 2 (dropout-only) for completeness, but treat Run 3 (weight-decay-only) as OPTIONAL/P2 —
        │           expected saved GPU-hours: ~21 if Run 3 is skipped.
        └─ IF Run 1 alone explains only a small fraction (WER stays >22%):
              └─ Run 2 (dropout-only) AND Run 3 (weight-decay-only) both become P0 — full sequential
                    decomposition needed to find the actual driver(s).
```

### GPU-hour estimates per branch

| Branch | Best-case (GPU-h) | Expected (GPU-h) | Worst-case (GPU-h) |
|---|---|---|---|
| Spike D (1-step smoke test) | ~0.02 | ~0.02 | ~0.02 (fails fast either way) |
| Adaptation-scope Branch 2 (broader LoRA) | ~21 | ~21 | ~21 (fixed-cost, not gated) |
| Adaptation-scope Branch 3 (full decoder FT) | 0 (Spike D fails, dropped) | ~21 (Spike D passes) | ~21 |
| Regularization Run 1 (SpecAugment-only) | ~21 | ~21 | ~21 (fixed-cost) |
| Regularization Run 2 (dropout-only) | ~21 | ~21 | ~21 (fixed-cost, always run) |
| Regularization Run 3 (weight-decay-only) | 0 (skipped, Run1 explains most) | ~10.5 (50% chance, illustrative) | ~21 (always run) |
| Decoding-fairness experiment (N-best+KenLM) | ~0 (reuses existing/new checkpoints, inference-only) | ~0 | ~0 |
| **Total P0 program** | **~63** | **~94.5** | **~105** |

---

## 5. Decoding Fairness

**Comparison options evaluated:**
1. W2V2 greedy vs. Canary native — feasible today, zero new work, but doesn't resolve the reviewer's core complaint (neither side has an external LM, so it's fair but incomplete as the *only* comparison).
2. W2V2+KenLM vs. Canary native — **this is the exact contamination reviewers flagged; do not use as the primary/headline comparison.**
3. W2V2+KenLM vs. Canary N-best+same KenLM — **CONFIRMED FEASIBLE this session** (Spike A/B). This is the lightest scientifically defensible fix: no ILME, no density-ratio estimation, no model surgery — just beam-search N-best decode (already supported natively) plus a rescoring pass with the exact same KenLM binary already built for W2V2.
4. ILME/density-ratio correction — **explicitly rejected as disproportionate.** It requires estimating `p_ILM(y)` by running the decoder with the acoustic/encoder contribution removed — nothing in the installed NeMo speechlm2 module exposes this, it would require custom forward-pass surgery, and it introduces its own new failure modes (subtracting a poorly-estimated internal-LM score can make results *worse*, not more comparable). Per your own instruction not to recommend it merely because the reviewer mentioned it: **it is not recommended.**

**Chosen design:** Report BOTH (1) and (3) as complementary results — (1) as the "no external LM either side" honest baseline (directly answers "is the comparison fair without any LM involved"), and (3) as the "matched external LM" result (directly answers "is the comparison fair once both sides get the same in-domain LM boost"). This dual-report approach is more defensible than picking one, costs zero additional GPU-hours (both reuse the same trained checkpoints), and directly neutralizes Reviewer 4Pd7's Major Concern #1 without the complexity or risk of ILME.

---

## 6. Adaptation-Scope Study

| Candidate | Question Answered | Control | Variable Changed | GPU Cost | Scientific Value | Reviewer Value |
|---|---|---|---|---|---|---|
| (a) Current baseline (LoRA q/v-only) | Reference point | — | — | 0 (reuse existing verified number, 23.32%) | Baseline | Required as the fixed reference |
| (b) Broader LoRA (q/v/k/o/MLP) | Is q/v-only under-parameterized? | (a) | LoRA target_modules only | ~21h | **High** — directly tests whether LoRA scope, not just presence, matters | **High** — directly answers Reviewer 4Pd7 Concern #2 |
| (c) Full decoder fine-tune | Is the frozen decoder itself the bottleneck? | (a) | Freeze/LoRA removed entirely for `llm.*` | ~21h, gated on Spike D | **Highest if feasible** — the paper's central untested claim | **Highest** — this is literally what the manuscript's interpretation rests on |
| Encoder LoRA / encoder+decoder LoRA | Does encoder adaptation matter? | (a) | Unfreeze/LoRA the FastConformer encoder | ~21h | Low-medium — tests a hypothesis the paper doesn't actually make (the paper's claim is about the *decoder*) | Low — not what reviewers asked about |
| Full model fine-tuning | Does unfreezing everything help? | (a) | Everything trainable | Almost certainly infeasible (2.87B full fp16 optimizer states on 11GB×4) | Low marginal value over (c) | Low — not requested by any reviewer |

**Minimum discriminative set: (a) [reused, no new run] + (b) + (c) [gated on Spike D].** Encoder-adaptation and full-model-fine-tuning are both **DROP** — they don't discriminate between the competing explanations the reviewers actually raised (decoder adaptation scope, not encoder adaptation), and full-model fine-tuning is very likely infeasible on this hardware regardless.

---

## 7. Regularization Study

**[VERIFIED FACT, from diff of `salm_uwb_atcc.yaml` vs. `salm_uwb_atcc_v3.yaml`]:** three variables changed simultaneously between the v1-equivalent baseline and v3: `lora_dropout` 0.01→0.1, SpecAugment none→(freq_masks=2, time_masks=10), `weight_decay` 1e-3→1e-2.

**Adaptive sequential design (not a full 2³=8-run factorial):**
1. **Reference (already have):** v1-equivalent, 23.32% WER.
2. **Run 1 — SpecAugment only.** Hypothesis: largest single contributor, since it acts directly on the audio input and directly addresses the "10.5 hours is not enough data variety" mechanism the manuscript itself proposes (Sec. VI.B).
3. **Decision gate:** if Run 1 alone recovers WER to within ~1-2pp of v3's 20.70%, dropout and weight_decay's marginal contributions are likely small — Run 3 (weight-decay-only) can be demoted to P2/optional, saving ~21 GPU-hours. Run 2 (dropout-only) is still worth running regardless, since it's cheap relative to the information gained about whether dropout contributes anything at all.
4. **Run 2 — dropout only.** Always run — completes the two most-likely-informative single-factor tests.
5. **Run 3 — weight-decay only.** Conditional per the Run-1 gate above.
6. **Optional Run 4 — best single factor + one more (not all three).** Only if Runs 1-3 together don't explain the full v1→v3 gap, to test for interaction effects with a single additional data point rather than a full factorial.

This is the **minimum sequential ablation** that isolates each of the three reviewer-named components while allowing early results to eliminate the least-likely-informative remaining run.

---

## 8. Data-Scale Study

**Verdict: NOT included in the P0 program.** Reasoning: the chosen scientific core (Section 2) is explicitly an adaptation-scope study, not a data-efficiency study. Adding a full data-scaling arm (even at 2-3 scales) would roughly double the compute budget without directly serving the chosen research question — it would answer Reviewer CXTt's question but at the cost of diluting the adaptation-scope study's focus, and split attention across two research questions in one paper is exactly the kind of unfocused framing Reviewer X1FA already criticized ("the main research question and take-away are not sufficiently focused"). **Recommendation: explicitly note this as future work in the redesigned paper's Limitations/Future Work section, do not attempt it in this program.** If pursued later, one corpus only (UWB-ATCC, larger and more realistic) at 2 scales (e.g., 25% and 100% — the two endpoints most informative for a linear-trend read, not a full 5-point curve) would be the minimum informative design.

---

## 9. Speaker-Independent Evaluation — Correctness Experiment, Not a Hyperparameter Search

**Exact train/test speakers [VERIFIED FACT, this session]:** `train_female`={gf1, zf1, zf2} (3,471 utt / 3.66h), `test_female`={zf3} (616 utt); `train_male`={sm1, sm2, sm3, sm4} (4,849 utt / 5.30h), `test_male`={gm1, gm2} (640 utt).

**Leakage checks [VERIFIED FACT]:** 0 recording overlap, 0 utterance-id overlap, both directions and cross-gender, via `comm -12` on `segments`/`text` files. LM leakage risk remains open (`ISS-005`) — the only trained ATCOSIM KenLM was built on the leaked full split; must not be reused here without retraining on `train_female`/`train_male` text only.

**Acoustic preprocessing:** identical pipeline to the main ATCOSIM splits (32kHz→16kHz via SoX), verified via `wav.scp` sample line inspection.

**Decoding protocol:** greedy (no LM) for the existing W2V2 EXP-007 results — matches the paper's own convention for the un-LM'd comparison.

**Per-speaker WER, pooled WER, uncertainty:** **NOT YET COMPUTED** for the per-speaker breakdown (gm1 vs. gm2 individually) — this is a zero-GPU-hour analysis task using EXP-007's existing hypothesis/reference outputs, and should be done before writing up the male-split result, since `test_male` pools 2 speakers and a large per-speaker difference would materially change how the 19.973% pooled number should be interpreted.

**Limitations to state explicitly in the paper:** `test_female` has **n=1** held-out speaker — this is a single-speaker WER, not a population estimate. `test_male` has **n=2** — allows a per-speaker comparison but still a very small sample for any claim of general speaker-independence.

**Should both models be evaluated under this protocol?** Currently only W2V2 has been run (EXP-007). Per Spike E's finding, **no gender-stratified lhotse cuts exist yet for Canary-Qwen** — building them is a near-zero-cost data-conversion prerequisite. Given the chosen research question (Section 2) centers on Canary-Qwen's adaptation scope, **recommend running the CHOSEN best adaptation-scope configuration (from Section 6, whichever wins) through this same speaker-independent protocol** as a capstone generalization check — not every adaptation-scope variant, just the winner. This keeps the speaker-independent evaluation as a confirmatory capstone, not a multiplier on the adaptation-scope grid.

---

## 10. Cross-Corpus Generalization

**Feasibility checklist:**
- **Sampling rate:** UWB-ATCC native 8kHz, ATCOSIM native 32kHz — both are resampled to 16kHz for Canary-Qwen already, but W2V2 currently consumes UWB-ATCC at 8kHz directly (per the manuscript, Sec. III.B: "Wav2Vec 2.0 accepted the original 8 kHz audio directly") — **a cross-corpus W2V2 run would require re-resampling UWB-ATCC to match ATCOSIM's convention or vice versa, a real, non-trivial pipeline change** not currently present in any script.
- **Transcript format:** not yet verified compatible at the normalization-rule level (acronym expansion, number-to-word conversion differ in tooling between the two data-prep scripts) — **OPEN QUESTION**, not yet checked.
- **Vocabulary:** ATCOSIM is scripted/repetitive; UWB-ATCC is broader/real communications — a model fine-tuned on one and tested on the other will hit substantial out-of-vocabulary and phrasing mismatch independent of any "domain shift" in the acoustic sense, conflating vocabulary mismatch with acoustic domain shift in any resulting WER number.
- **Channel/acoustic differences:** real telephone-quality noisy audio (UWB-ATCC) vs. clean close-talk headset (ATCOSIM) — a large, confound-inducing difference on its own.
- **LM leakage:** any KenLM used for cross-corpus decoding would need to be corpus-matched to the *test* corpus, not the training corpus, and clearly labeled as such.
- **Does fine-tuning invalidate "unseen domain"?** Yes — a model fine-tuned on corpus A and tested on corpus B is testing fine-tuned cross-corpus transfer, not zero-shot unseen-domain generalization; correct terminology is "cross-corpus transfer after in-domain fine-tuning."

**Verdict: methodologically weak given the confound stack above (sample-rate pipeline mismatch for W2V2 specifically, unverified transcript compatibility, vocabulary/channel confound). NOT run in the P0 program.** If pursued at all, only **UWB-ATCC→ATCOSIM** (not the reverse) is worth attempting, since ATCOSIM's narrower vocabulary makes a UWB-ATCC-trained model's transfer more interpretable (a broad-vocabulary model meeting a narrow-vocabulary test set) than the reverse (a narrow-vocabulary model meeting a broad, unfamiliar test set, which would likely just fail uninformatively). **Recommend explicitly dropping cross-corpus from this program and stating in the paper's Limitations that same-corpus disjoint-speaker evaluation (Section 9), not cross-corpus transfer, is the generalization evidence provided** — this is the honest, narrower claim the available data and pipeline actually support.

---

## 11. Full Retraining Strategy

Given checkpoints are lost, everything technically needs retraining — but not everything needs retraining *for this paper*.

**A. Baseline reproduction (P2/optional):** UWB-ATCC W2V2-large, ATCOSIM W2V2-large, Canary-Qwen UWB-ATCC LoRA-only. All three already have provenance-verified numbers from before the checkpoint loss; re-running only buys a fresh-seed confirmatory point. Skip unless time/compute remains after P0.

**B. Necessary scientific controls (P0):** the v1-equivalent Canary-Qwen baseline (23.32%) is the fixed reference point for both the adaptation-scope study (Section 6) and the regularization study (Section 7) — **this needs to exist as an actually-held checkpoint** (not just a cited historical number) so that N-best decoding (Section 5) and per-utterance comparison against the new ablation runs can be done consistently. **Recommend re-running this one baseline fresh, under the current verified environment, as the true Run-0 control for everything else** — this is the one "reproduction" run that isn't purely confirmatory, since without a live checkpoint the entire adaptation-scope and regularization comparison has no common control.

**C. Reviewer-driven experiments (P0/P1, this program's core):** Section 6 (adaptation scope), Section 7 (regularization), Section 5 (decoding fairness), Section 9's capstone (winning config on speaker-independent split).

**D. Optional confirmatory runs (P1/P2):** a second-seed run of whichever configuration becomes the paper's headline result (Stage 5).

**Smallest clean baseline set:** one freshly-trained Canary-Qwen v1-equivalent checkpoint (item B) — everything else in the program is a genuinely new ablation relative to it, not a reproduction.

---

## 12. Hardware/Software Optimization Per Experiment

| Experiment | GPU count | VRAM | DDP/FSDP | Wall-clock | GPU-hours | CPU/dataloader | Disk |
|---|---|---|---|---|---|---|---|
| Fresh v1-equivalent control run | 4 | fits today (LoRA, verified) | FSDP (unchanged — [[DEC-004]]) | ~5.3h | ~21 | `num_workers: 1` (current config; not a bottleneck at this data scale) | ~1-2GB checkpoint |
| Spike D (1-step smoke) | 4 (must match production FSDP topology to be meaningful) | unknown — this is what's being tested | FSDP | <5 min | ~0.02 | n/a | negligible |
| Adaptation-scope Branch 2 (broader LoRA) | 4 | plausibly higher than current (more trainable params) but still LoRA-scale, low risk | FSDP | ~5.3h | ~21 | unchanged | ~1-2GB |
| Adaptation-scope Branch 3 (full decoder FT) | 4 | **at risk — this is exactly what Spike D tests** | FSDP | ~5.3h if it fits | ~21 | unchanged | larger checkpoint (~3.4GB for fp16 Qwen3-1.7B decoder weights alone) |
| Regularization Runs 1-3 | 4 each | fits today (same as baseline) | FSDP | ~5.3h each | ~21 each | unchanged | ~1-2GB each |
| Decoding-fairness (N-best decode + KenLM rescore) | 1 (inference-only, no training) | small — inference batch, not training-scale | none needed | minutes | ~0 (rounds to negligible) | n/a | negligible |
| Speaker-independent capstone (winning config, Canary-Qwen) | 4 | same as its adaptation-scope branch | FSDP | ~5.3h | ~21 | unchanged | ~1-2GB |

**None of the training experiments should reduce below 4 GPUs / FSDP** — per your own instruction, changing the production DDP/FSDP mechanism merely for convenience would invalidate the comparison with the existing baseline numbers (all of which were measured under this exact 4-GPU FSDP topology). **The only experiment appropriately run on fewer GPUs is Spike D's 1-step smoke test**, and even that should use the full 4-GPU FSDP topology since the VRAM question being tested is specifically about *this* sharding configuration, not a general single-GPU question.

**Early-stopping applies to:** Regularization Runs 1-3 (per Section 7's decision gate) and the overall adaptation-scope study (per Section 4's tree) — not to individual runs' own step counts, which should remain matched to the existing 10,000-step convention for direct comparability with historical numbers (changing step count mid-comparison would introduce a new confound).

---

## 13. Statistical Plan

**Primary comparisons requiring bootstrap CIs and (where paired on the same test set) paired bootstrap comparison:**
- Adaptation-scope Branch 2 vs. Branch 1 (baseline) — same test set, paired.
- Adaptation-scope Branch 3 vs. Branch 1 — same test set, paired (if Branch 3 runs).
- Regularization Run 1/2/3 vs. Run 0 baseline — same test set, paired, for each.
- W2V2+KenLM vs. Canary N-best+KenLM (Section 5's fairness comparison) — same test set, paired.

**Per-speaker analysis required for:** the ATCOSIM speaker-independent capstone (both `test_male`'s gm1/gm2 breakdown and, trivially, `test_female`'s single-speaker result stated as n=1).

**Repeated seeds:** only for the final chosen headline configuration (Stage 5), not across the board — matches the "repeated seeds only when scientifically justified" instruction and keeps the compute budget from ballooning.

**Learning curves as figures, not tables** — resolves Reviewer 4Pd7's presentation concern directly, applies to the adaptation-scope and regularization comparisons.

**Explicitly secondary/not required:** ANOVA across every factor, effect-size statistics beyond the paired-bootstrap comparisons above, data-scale curves (Section 8 dropped), cross-corpus statistics (Section 10 dropped).

---

## 14. Paper Contribution Under Three Outcomes

**BEST CASE — full decoder fine-tuning (Branch 3) closes most of the gap to W2V2:**
Supported by: Branch 3 WER within ~2-3pp of W2V2's 14.54%. Contribution: "Adaptation scope, not architecture, is the dominant factor limiting Canary-Qwen's ATC performance — full decoder fine-tuning closes N percentage points of the gap that LoRA-only adaptation leaves open." This directly validates and *properly supports* (unlike the current manuscript) the frozen-decoder-bottleneck interpretation.

**NEUTRAL CASE — W2V2 still wins even after full decoder fine-tuning, but the gap narrows meaningfully:**
Supported by: Branch 3 WER better than Branch 1/2 but still several points behind W2V2. Contribution: "Adaptation scope substantially affects Canary-Qwen's ATC performance and explains part, but not all, of the gap to a fully fine-tuned CTC baseline — the residual gap implicates architecture, pretraining domain, or optimization dynamics beyond adaptation scope alone." Still a genuine, publishable, honest finding — arguably the *most* interesting outcome, since it partially confirms and partially refutes the original hypothesis with evidence either way.

**NEGATIVE CASE — full decoder fine-tuning barely changes WER, or Spike D fails and only LoRA-scope results exist:**
Supported by: Branch 3 (if run) shows minimal improvement over Branch 1, or Branch 3 couldn't be run at all. Contribution: "Decoder adaptation scope, within what is feasible on commodity hardware, does not substantially close the gap to a fully fine-tuned CTC baseline for this domain — a hybrid SALM's frozen general-text pretraining, not merely its parameter-efficient adaptation, is implicated." This is a real negative result and a legitimate contribution (it directly refutes the original manuscript's central interpretive claim with actual evidence, rather than the original's untested assertion) — the redesign should be written to accommodate this outcome as gracefully as the others, not to presuppose the BEST CASE.

**The paper remains scientifically useful under all three outcomes** because the research question (Section 2) was framed around "how does adaptation scope affect X" rather than "adaptation scope will close the gap" — the latter framing would only be publishable in the BEST CASE.

---

## 15. Final Execution Plan

### STAGE 0 — Repository/data/environment audit
**OBJECTIVE:** Confirm all prerequisites exist before spending any compute.
**EXPERIMENTS:** None (analysis only) — largely complete via this session and the prior session's `AUD-004`/`VAL-004`/`VAL-005`.
**DEPENDENCIES:** None.
**GO/NO-GO:** Already GO — repo, upstream code, hardware, and manuscript have all been inspected.
**EXPECTED GPU-HOURS:** 0. **WALL-CLOCK:** Done. **INFORMATION GAIN:** High (this is what surfaced the encoder-unfrozen provenance issue, the N-best feasibility, and the missing gender-stratified Canary-Qwen lhotse cuts).

### STAGE 1 — Zero-cost feasibility spikes
**OBJECTIVE:** Resolve Spikes A/B/C by inspection (done) and confirm A/B empirically if desired; run Spike D's 1-step smoke test (the one spike requiring actual GPU time and your explicit approval, since it launches a process).
**EXPERIMENTS:** Spike D (1-step smoke test, ~0.02 GPU-hours); optional empirical confirmation of Spike A/B.
**DEPENDENCIES:** None.
**GO/NO-GO:** GO on Spike D determines whether Adaptation-Scope Branch 3 exists in Stage 3 at all.
**EXPECTED GPU-HOURS:** ~0.02. **WALL-CLOCK:** <30 min. **INFORMATION GAIN:** Very high relative to cost — this single cheap test determines ~21 GPU-hours' worth of downstream scheduling.

### STAGE 2 — Minimal baselines
**OBJECTIVE:** Establish the one live control checkpoint needed for everything downstream (Section 11.B).
**EXPERIMENTS:** Fresh Canary-Qwen v1-equivalent (LoRA q/v-only, no added regularization) run on UWB-ATCC.
**DEPENDENCIES:** None (can run in parallel with Stage 1's Spike D, since they don't share GPU time if scheduled sequentially on this 4-GPU machine — but this machine has exactly 4 GPUs total, so Stage 1 and Stage 2 are effectively sequential, not parallel, in practice).
**GO/NO-GO:** GO unconditionally — this checkpoint is needed regardless of any other stage's outcome.
**EXPECTED GPU-HOURS:** ~21. **WALL-CLOCK:** ~5.3h. **INFORMATION GAIN:** Enables everything else; without it, no paired-bootstrap comparison (Section 13) is possible against a real checkpoint.

### STAGE 3 — Decisive ablations
**OBJECTIVE:** Run the adaptation-scope and regularization branches per Section 4's decision tree.
**EXPERIMENTS:** Adaptation-scope Branch 2 (broader LoRA, always); Branch 3 (full decoder FT, gated on Stage 1's Spike D); Regularization Runs 1-3 (Run 1/2 always, Run 3 gated on Run 1's result).
**DEPENDENCIES:** Stage 1 (Spike D result) and Stage 2 (control checkpoint, for consistent comparison).
**GO/NO-GO:** Each run's own decision gate, per Section 4.
**EXPECTED GPU-HOURS:** ~63-84 (see Section 4's table). **WALL-CLOCK:** ~3-4.5 days sequential on this 4-GPU machine. **INFORMATION GAIN:** This is the core of the paper — highest information density of any stage.

### STAGE 4 — Generalization evaluation
**OBJECTIVE:** Run the decoding-fairness comparison (Section 5) and the speaker-independent capstone (Section 9) using whichever configuration Stage 3 identifies as the winning adaptation-scope setup.
**EXPERIMENTS:** N-best+KenLM decode/rescore (inference-only); winning-config Canary-Qwen run on the gender-stratified ATCOSIM splits (requires building the missing lhotse cuts first, per Spike E — a near-zero-cost data-conversion step, not training).
**DEPENDENCIES:** Stage 3's winning configuration; Spike E's lhotse-cuts prerequisite.
**GO/NO-GO:** GO unconditionally once Stage 3 has a winner.
**EXPECTED GPU-HOURS:** ~21 (one more training run, the capstone) + ~0 (inference-only decoding-fairness). **WALL-CLOCK:** ~5.3h + minutes. **INFORMATION GAIN:** High — this is what makes the paper's generalization claim (narrowly, correctly scoped to speaker generalization) defensible.

### STAGE 5 — Confirmatory runs
**OBJECTIVE:** One second-seed run of the final headline configuration, for a minimal seed-variance estimate.
**EXPERIMENTS:** One re-seed of whichever Stage 3 branch becomes the paper's headline result.
**DEPENDENCIES:** Stage 3 complete.
**GO/NO-GO:** Optional (P1) — run only if time/compute remains.
**EXPECTED GPU-HOURS:** ~21. **WALL-CLOCK:** ~5.3h. **INFORMATION GAIN:** Modest (a single additional seed doesn't establish a real variance estimate, but it does guard against the headline result being a one-off fluke).

### STAGE 6 — Manuscript reconstruction
**OBJECTIVE:** Rewrite the paper per `IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` Section 15, incorporating whichever of the three outcomes (Section 14) actually occurred, updating the Acknowledgements/AI-disclosure section to reflect Claude's direct execution role in EXP-007 and this program (per the reconciliation finding from the prior session).
**EXPERIMENTS:** None — writing only.
**DEPENDENCIES:** All experimental stages complete.
**GO/NO-GO:** N/A.
**EXPECTED GPU-HOURS:** 0. **WALL-CLOCK:** Parallelizable with Stage 5 (writing can start once Stage 4 completes, while Stage 5's confirmatory run is in progress). **INFORMATION GAIN:** N/A (synthesis, not evidence generation).

---

## 16. Master Experiment Table

| EXP-ID | Reviewer Concern | Question | Model | Data | Adaptation | Decoding | GPU Count | Expected Hours | Stop Criteria | Dependencies | Status | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1-D | Frozen-decoder hypothesis untested | Does full decoder FT fit in VRAM? | Canary-Qwen | UWB-ATCC (1 batch) | Full decoder | N/A | 4 | 0.02 | OOM at step 1 | None | PROPOSED | **P0** |
| S2-CTRL | (enables all comparisons) | Fresh, live v1-equiv control checkpoint | Canary-Qwen | UWB-ATCC | LoRA q/v-only | native | 4 | 21 | Full 10k steps or clear divergence | None | PROPOSED | **P0** |
| S3-B2 | Decoder-LoRA scope untested (4Pd7 #2) | Does broader LoRA target set improve WER? | Canary-Qwen | UWB-ATCC | LoRA (7 proj types) | native | 4 | 21 | Full 10k steps or clear divergence | S1-D (config sanity only) | PROPOSED | **P0** |
| S3-B3 | Frozen-decoder hypothesis (4Pd7 #2, CXTt) | Does full decoder FT close the gap? | Canary-Qwen | UWB-ATCC | Full decoder | native | 4 | 21 | Full 10k steps or clear divergence | S1-D = GO | PROPOSED | **P0 if S1-D passes, else DROP** |
| S3-R1 | Confounded regularization (CXTt, BzrK) | Does SpecAugment alone recover the plateau? | Canary-Qwen | UWB-ATCC | LoRA q/v-only + SpecAugment | native | 4 | 21 | Full 10k steps or clear divergence | S2-CTRL | PROPOSED | **P0** |
| S3-R2 | Confounded regularization | Does dropout alone recover the plateau? | Canary-Qwen | UWB-ATCC | LoRA q/v-only + dropout=0.1 | native | 4 | 21 | Full 10k steps or clear divergence | S2-CTRL | PROPOSED | **P0** |
| S3-R3 | Confounded regularization | Does weight_decay alone recover the plateau? | Canary-Qwen | UWB-ATCC | LoRA q/v-only + wd=1e-2 | native | 4 | 21 | Full 10k steps or clear divergence | S3-R1 result | PROPOSED | **P1, conditional on S3-R1** |
| S4-FAIR | Apples-to-oranges LM comparison (4Pd7 #1) | Is W2V2+KenLM vs. Canary native unfair, and does N-best+KenLM fix it? | Both | UWB-ATCC test | (uses existing/S3 checkpoints) | greedy/KenLM/N-best+KenLM | 1 (inference) | ~0 | N/A (analysis) | S3 winning config | PROPOSED | **P0** |
| S4-SPK | No speaker-independent eval for Canary-Qwen (X1FA, CXTt, BzrK) | Does the winning adaptation-scope config generalize to held-out ATCOSIM speakers? | Canary-Qwen | ATCOSIM gender-stratified | (S3 winner's scope) | native | 4 | 21 | Full steps or clear divergence | S3 winner + new lhotse cuts (Spike E) | PROPOSED | **P0** |
| S5-RESEED | Single-seed result robustness | Does the headline result hold under a second seed? | Canary-Qwen | UWB-ATCC | (headline config) | native | 4 | 21 | Full steps | Stage 3 complete | PROPOSED | **P1** |
| — | Data-scale question (CXTt) | How much data does Canary-Qwen need? | — | — | — | — | — | — | — | — | **DROP** (Section 8) | **DROP** |
| — | Cross-corpus "unseen domain" (X1FA, CXTt, BzrK) | Does fine-tuning transfer across corpora? | — | — | — | — | — | — | — | — | **DROP** (Section 10) | **DROP** |
| — | Encoder-unfrozen re-verification | Was the encoder actually unfrozen historically? | — | — | — | — | — | — | — | — | **DROP unless independently prioritized** ([[ISS-007]]) | **P2** |
| — | Baseline reproduction (W2V2, Canary LoRA-only) | Do historical numbers still reproduce? | — | — | — | — | — | — | — | — | Already have provenance-verified numbers | **P2** |

---

## 17. Compute Budget

| Program | GPU-hours | Wall-clock (4-GPU sequential) |
|---|---|---|
| **MINIMUM** (Spike D fails → drop S3-B3; Run 1 explains most of reg. gap → drop S3-R3) | **~63** | **~3.4 days** |
| **EXPECTED** (Spike D passes with 50% assumed chance of needing S3-R3) | **~94.5** | **~5.1 days** |
| **MAXIMUM** (Spike D passes, S3-B3 runs; S3-R3 also runs; S5 confirmatory reseed runs) | **~126** | **~6.75 days** |

**Experiments responsible for most of the compute:** S3-B3 (full decoder FT, ~21h, entirely conditional) and the three-way regularization ablation (S3-R1/R2/R3, up to ~63h combined) together account for 60-70% of the maximum program. **S2-CTRL (the fresh control checkpoint) is a fixed, non-negotiable ~21h cost** that every other comparison depends on — it is the single highest-leverage individual run in the program, since without it nothing else has a valid, freshly-verified reference point.

**Cost vs. information gained:** the minimum program (~63 GPU-hours, ~3.4 days) already answers the two CRITICAL SCIENTIFIC reviewer concerns (decoder-LoRA scope, confounded regularization) plus the decoding-fairness fix — all at effectively zero marginal GPU cost beyond the training runs already needed for the scope/regularization questions. The jump from minimum to maximum (~63→126h) buys: the full-decoder-FT data point (highest single-experiment value in the whole program, since it directly tests the manuscript's central claim), the third regularization component, and a confirmatory reseed — all individually lower-value than the minimum program's core, but not wasteful either.

---

## 18. Research-Memory Updates

Per Part 18 of the request, this section documents what was written to `research_log/` this session (not re-derived here — see the actual files for full text):

- **`ISS-008`** (new): missing gender-stratified Canary-Qwen lhotse cuts — a concrete near-zero-cost prerequisite discovered via Spike E, blocking Section 9's capstone experiment until built.
- **`DEC-006`** (new): decision to drop data-scale (Section 8) and cross-corpus (Section 10) studies from the P0/P1 program, with reasoning preserved.
- **`EXP-010`** (new): the full execution-ready program itself (this document), status PROPOSED, linked to `DEC-005`/`DEC-006`/`ISS-007`/`ISS-008`.
- **`VAL-008`** (new): the N-best generation feasibility finding (Spike A/B resolution via source-code inspection), classified VERIFIED FACT.

All prior historical conclusions (`ISS-001` through `ISS-007`, `DEC-001` through `DEC-005`, `EXP-001` through `EXP-009`, `VAL-001` through `VAL-007`, `AUD-001` through `AUD-004`) remain intact and unmodified — nothing was deleted or silently rewritten. See the actual files for the full text of each new record.

---

## 19. Executive One-Page Plan

**1. Exact scientific question:** Within Canary-Qwen (architecture/scale/pretraining held fixed), how does decoder adaptation scope (LoRA q/v-only → broader LoRA → full decoder fine-tune, if VRAM allows) affect ATC WER, and — under a decoding-fairness-corrected comparison — how much of the residual gap to a fully fine-tuned W2V2-large CTC baseline does adaptation scope explain? This explicitly does NOT ask "which architecture is better" and does NOT claim "unseen domain" generalization beyond a narrowly-scoped speaker-independence result.

**2. The 3-5 most important experiments:** (1) **Spike D** — the 1-step VRAM feasibility smoke test, because it's nearly free and determines ~21 GPU-hours of downstream scheduling; (2) **S2-CTRL** — the fresh live control checkpoint, because everything else depends on it existing; (3) **S3-B3 (full decoder fine-tune)** if Spike D passes — the single experiment that most directly tests the manuscript's own central, currently-unsupported claim; (4) **S3-R1 (SpecAugment-only ablation)** — the highest-value single regularization-decomposition run; (5) **S4-FAIR (N-best+KenLM decoding-fairness comparison)** — resolves the reviewers' most universally-cited technical complaint at near-zero additional cost.

**3. Experiments deliberately dropped:** the full data-scale study (Section 8 — would double compute without serving the chosen research question); cross-corpus generalization (Section 10 — too many stacked confounds, including a real sample-rate pipeline mismatch for W2V2, to be interpretable); the full regularization 2³ factorial (replaced by the adaptive sequential design); encoder-adaptation and full-model-fine-tuning branches of the adaptation-scope study (don't discriminate between the hypotheses reviewers actually raised); baseline reproduction of already-provenance-verified historical numbers (W2V2 both corpora, Canary LoRA-only) — demoted to optional/P2.

**4. Go/no-go gates:** Spike D (OOM at step 1 → drop S3-B3 entirely); S3-R1's result (recovers most of the v1→v3 gap → demote S3-R3 to optional); Spike A/B (already resolved GO via code inspection, no further gate needed).

**5. Minimum expected GPU-hour budget:** ~63 GPU-hours (~3.4 wall-clock days), if Spike D fails and S3-R1 alone explains most of the regularization gap.

**6. Expected maximum budget:** ~126 GPU-hours (~6.75 wall-clock days), if Spike D passes, all three regularization components are needed, and one confirmatory reseed run is included.

**7. Result that would most strongly change the paper:** Spike D's pass/fail outcome, and — if it passes — whether S3-B3's WER closes to within a few points of W2V2 (BEST CASE narrative) or barely moves (NEGATIVE CASE narrative, which is still a legitimate, honest contribution but tells a materially different story).

**8. Biggest remaining scientific risk:** Spike D failing (OOM) would remove the single experiment that most directly tests the manuscript's central interpretive claim, leaving the adaptation-scope study as a LoRA-scope-only comparison — still valid and reviewer-responsive, but a narrower contribution than the full three-point comparison would provide. Separately, `ISS-007` (the encoder-unfrozen provenance gap) can never be fully resolved since the original checkpoints are gone — the redesigned paper must either drop that historical ablation or cite it with an explicit provenance caveat.

**9. What to do first tomorrow morning:** Run **Spike D** (the 1-step VRAM smoke test) — it costs ~0.02 GPU-hours, takes under 30 minutes, and its outcome determines whether the single highest-value experiment in the entire program (S3-B3, full decoder fine-tuning) is even on the table. This requires your explicit approval to launch, since it does start a (very short) training process. Immediately after, start **S2-CTRL** (the fresh control checkpoint) regardless of Spike D's outcome, since it's needed either way and is the longest single fixed-cost item blocking everything downstream.
