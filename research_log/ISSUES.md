# Issues and Risks

## ISS-001 — ATCOSIM gender-subset evaluation has train/test speaker leakage

Date: 2026-09-06 (recovered; original discovery during Phase 4 ATCOSIM work)
Status: RESOLVED (by discarding the results; a valid re-run remains open, see DEC-001)
Severity: HIGH (invalidates a reported scientific result if reused)

Description: The ATCOSIM full-corpus model was trained on a random 80/20 split spanning all 10 speakers. The gender-based test subsets (test_female = zf3, test_male = gm1/gm2) are speaker-filtered views of that same corpus, so ~80% of each subset speaker's utterances were already seen in training. Any WER computed this way is not a speaker-independent measurement.

Evidence: models/w2v2/docs/PROGRESS_ATCOSIM.md, "Phase 4 - Gender Experiments (Speaker-Independent Eval) [ATTEMPTED — DATA LEAKAGE FOUND]"; see [[AUD-001]].

Impact: The numbers 0.86% WER (test_female) and 0.01% WER (test_male), and their eval_model.py counterparts 86%/1.23%, are unusable as evidence of speaker-independent generalization. They must not be cited as ATCOSIM's "true" WER in any comparison.

Mitigation: Results were explicitly marked as discarded/not trustworthy in-repo (PROGRESS_ATCOSIM.md) rather than deleted, preserving the audit trail.

Resolution: A corrected design was decided: train separate models on `train_female`/`train_male` splits and evaluate on the disjoint held-out speaker sets. This corrected experiment was documented in DEC-001/SUMMARY.md "Pending / Next Steps" as ATCOSIM Phase 4 (valid) and, per git log, is not yet executed as of the latest commit (054bd54) in this repo — status remains PROPOSED/PENDING for the corrected run.

Related Records: [[AUD-001]], [[DEC-001]]

---

## ISS-002 — eval_model.py hypothesis-file bug when no LM is supplied

Date: 2026-09-06 (recovered)
Status: RESOLVED (root cause understood; workaround documented, not a code fix)
Severity: LOW (cosmetic — printed summary WER is correct; only the saved hypothesis file is misleading)

Description: In the paper's `eval_model.py` (from idiap/w2v2-air-traffic), when no language model is provided, the script sets `pred_str_ctc_lm` equal to the reference text, which makes the saved hypothesis output file look like perfect predictions. The actual printed WER is computed correctly from greedy `pred_str` vs. the decoded labels.

Evidence: SUMMARY.md "Key Technical Findings" item 4; corroborated in AUD-001 (gender eval finding: "eval_model.py also has a bug (uses CTC-decoded labels as reference) that inflates WER vs the custom script").

Impact: Anyone reading only the saved hypothesis file (not the printed summary metric) for a no-LM run would wrongly conclude the model is perfect. Also causes eval_model.py and a custom scoring script to disagree on WER for the same run (e.g., ATCOSIM gender eval: 86% vs 0.86%).

Mitigation: Documented explicitly so future readers trust the printed summary WER, not the hypothesis file, for no-LM runs.

Resolution: No code change was made to eval_model.py (external, third-party repo `idiap/w2v2-air-traffic`, not vendored here). Treated as a known quirk to work around.

Related Records: [[AUD-001]]

---

## ISS-003 — fp16 training NaN with small AdamW epsilon on Canary-Qwen (NeMo FSDP)

Date: 2026-09-06 (recovered)
Status: RESOLVED
Severity: MEDIUM (caused a full training run to fail with NaN, wasting GPU time)

Description: Canary-Qwen "Research-optimized" Run 4 (lr=3e-5, r=64, 2500 steps, fp16-true) hit NaN loss at step 1500. Root cause identified as AdamW epsilon too small for fp16-true precision on the RTX 2080 Ti; the research-suggested eps=1e-6 was still too small.

Evidence: models/canary-qwen/docs/PROGRESS.md, "Run 4: Research-optimized" and "Known Issues & Fixes" ("fp16-true + eps=1e-8 causes NaN: use eps=1e-4"); REPLICATION_GUIDE.md Part 3 known-issues table ("fp16 + AdamW eps=1e-8 causes NaN → Set eps: 1e-4 in optimizer config").

Impact: Run 4 could not be evaluated at all — additionally compounded by a separate LoRA-rank mismatch (r=64 checkpoint vs pretrained r=128 model), so Run 4 produced no usable result either way.

Mitigation / Resolution: Standardized on AdamW `eps=1e-4` for all fp16-true Canary-Qwen training runs going forward (used in Runs 1, 2, 3, 6/v3). Also standardized on LoRA r=128 to match the pretrained checkpoint's rank.

Related Records: [[ENV-003]], [[EXP-004]]

---

## ISS-004 — Stale trainable-param percentage in finetuned_results_unfrozen.json

Date: 2026-09-06 (recovered during this audit)
Status: **RESOLVED (2026-09-08)**
Severity: LOW

Description: `models/canary-qwen/docs/finetuned_results_unfrozen.json` (renamed to `finetuned_results_v2.json` as part of [[DEC-008]]'s v1/v2/v3 rename) stored `"params_trained_pct": 32.8`, the pre-correction value. Commit 054bd54 corrected the same figure (838.8M / 2,870M = 29.2%) everywhere in Markdown docs (REPLICATION_GUIDE.md, models/canary-qwen/docs/PROGRESS.md, shared/model_comparison.md) but did not touch this JSON file. `AUD-004` also found the same stale value in `train_canary_unfrozen.sh`'s header comment.

Evidence: `cat models/canary-qwen/docs/finetuned_results_unfrozen.json` → `"params_trained_pct": 32.8`; contrast with [[AUD-003]] / commit 054bd54.

Impact: Any future script or reader that consumes the JSON file directly (rather than the Markdown docs) would pick up the superseded 32.8% figure.

Mitigation/Resolution: Fixed 2026-09-08 while the file was already being touched for the v1/v2/v3 rename ([[DEC-008]]) — `params_trained_pct` corrected to 29.2 in `finetuned_results_v2.json`. The corresponding script comment (`train_canary_v2.sh`, formerly `train_canary_unfrozen.sh`) was also rewritten and no longer states a trainable-param percentage in a way that could go stale the same way.

Related Records: [[AUD-003]], [[AUD-004]], [[DEC-008]]

---

## ISS-005 — Existing ATCOSIM 4-gram KenLM was trained on the leaked main split; must not be used for speaker-independent decoding

Date: 2026-09-07
Status: OPEN (documented only — no code/script/model change made)
Severity: HIGH (would silently reintroduce train/test leakage into EXP-007's speaker-independent eval if reused)

Description: `experiments/data/atcosim_corpus/train/lm/atcosim_corpus_4g.binary` was built (via `src/run_train_kenlm.sh` → `src/train_kenlm.py`) from `experiments/data/atcosim_corpus/train/text` — the same leaked random 80/20 split identified in [[ISS-001]] (contains all 10 speakers, not a speaker-disjoint split). Verified: `comm -12` between this LM's training text utterance IDs and `test_female`/`test_male` utterance IDs shows 233/616 test_female utterances and all 640/640 test_male utterances already present in the LM's training text.

Evidence: `experiments/data/atcosim_corpus/train/utt2spk` contains all 10 speakers (zf3, gm1, gm2 included); `comm -12 <(cut -d' ' -f1 train/text|sort -u) <(cut -d' ' -f1 test_female/text|sort -u)` → 233; same against `test_male/text` → 640 (100% of test_male).

Impact: This LM is not itself invoked by EXP-007's current training/eval command (`run_asr_fine_tuning.sh` passes no `--lm`/language-model flag — EXP-007 decodes greedily), so it does NOT block the currently-planned EXP-007 launch. But if any future step fuses this LM with test_female/test_male decoding (e.g. via `eval_model.py` + PyCTCdecode), it would reintroduce ISS-001-style leakage through the language model even with a clean, speaker-disjoint acoustic model — silently invalidating an otherwise-valid speaker-independent result.

Mitigation: None applied yet. A leakage-free 4-gram LM for gender-subset eval would need to be trained separately from `train_female`/`train_male` text only (or omitted, keeping greedy decoding as EXP-007 currently does).

Resolution: Not resolved — flagged for awareness before any future LM-fusion step is added to EXP-007 or its successors. No script, model, or result file was modified to produce this record.

Related Records: [[ISS-001]], [[EXP-007]], [[VAL-005]]

---

## ISS-007 — Canary-Qwen "encoder unfrozen" config does not match the manuscript's claim for that ablation

Date: 2026-09-07
Status: OPEN (documented only — checkpoints lost, cannot be independently re-verified)
Severity: HIGH (a manuscript-cited result cannot currently be reproduced or verified)

Description: The submitted manuscript (`Final_Draft.md`, Table I / Sec. VI.B) reports a Canary-Qwen UWB-ATCC ablation with the encoder unfrozen: 838.8M/2,870M (29.2%) trainable params, WER 23.82%, with a distinct learning-curve trajectory in Table III. The GitHub-committed config file that should document this run, `models/canary-qwen/scripts/salm_uwb_atcc_unfrozen.yaml`, is MD5-identical to the frozen baseline `salm_uwb_atcc.yaml` — both currently freeze `^perception\.encoder\..+$`, meaning the committed "unfrozen" config does not actually unfreeze the encoder.

Evidence: `md5sum salm_uwb_atcc.yaml salm_uwb_atcc_unfrozen.yaml` → identical hash `8d3cd6e7a6dafa03ded133d46f7ca42c`; manuscript Table I/III and Sec. VI.B text describing the 838.8M-param unfrozen run.

Impact: The 23.82% WER / 838.8M-param result cannot currently be reproduced from the repository. Original checkpoints are lost (per user statement), so the claim cannot be independently verified against ground truth either. The result's internal plausibility (a distinct, non-trivial learning-curve trajectory) argues against outright fabrication, but this is a judgment call, not a verified fact.

Mitigation: **UPDATE (2026-09-08):** the WER result itself (23.82%) is now independently verified via real inference against the HuggingFace-hosted model — see [[VAL-012]]. The config-provenance problem is confirmed to be a **third independent instance** of the same pattern: the `training_config.yaml` uploaded alongside that same HF model is *also* byte-identical to the v1/baseline config (encoder shown frozen), despite the model measurably, reproducibly differing from v1. The committed local config was renamed `salm_uwb_atcc_v2.yaml` (2026-09-08) with an explicit header comment documenting this unresolved provenance gap, so future users aren't misled into thinking it reproduces the result.

Resolution: Partially resolved — the WER result is verified genuine; the exact hyperparameters that produced it remain unrecoverable from any file (local or HuggingFace) checked so far. Config file renamed and annotated, not fixed (cannot fix what isn't known).

Related Records: [[ISS-004]], [[AUD-004]], [[DEC-005]], [[VAL-012]]

---

## ISS-008 — No gender-stratified Canary-Qwen lhotse cuts exist yet for ATCOSIM speaker-independent evaluation

Date: 2026-09-07
Status: OPEN (documented only, not fixed — a data-conversion prerequisite, not a training action)

Description: `~/w2v2-air-traffic/experiments/data/atcosim_corpus/lhotse/` contains only `atcosim_train_cuts.jsonl` and `atcosim_test_cuts.jsonl` (the full, leaked-split corpus). No `train_female`/`train_male`/`test_female`/`test_male` lhotse cuts exist. W2V2's speaker-independent evaluation ([[EXP-007]]) used the Kaldi-format gender splits directly and did not need lhotse cuts; Canary-Qwen's NeMo/Lhotse pipeline requires them.

Evidence: `find ~/w2v2-air-traffic/experiments/data/atcosim_corpus/lhotse -type f` lists only the two full-corpus files.

Impact: Before any Canary-Qwen speaker-independent run (`research_report/FINAL_RESEARCH_PROGRAM.md` Section 9/Stage 4's "S4-SPK" experiment) can be scheduled, the four gender-specific lhotse cut files must be built from the existing Kaldi-format `train_female`/`train_male`/`test_female`/`test_male` directories — a near-zero-cost data-conversion step (reusing the existing `prepare_atcosim_lhotse.py`/`convert_manifests_to_lhotse.py` pattern already used for the full corpus), not a training action.

Mitigation: None applied yet — flagged as a prerequisite for Stage 4 of the execution plan.

Resolution: Not resolved.

Related Records: [[EXP-007]], [[EXP-010]], [[DEC-005]]

---

## ISS-009 — A real Canary-Qwen v3 UWB-ATCC checkpoint survives at ~/canary-ft/experiments/checkpoints/, contradicting "all checkpoints lost," and its config disagrees with the committed/cited one

Date: 2026-09-07
Status: OPEN (documented only, nothing deleted or modified)

Description: During Stage 2 preflight, `~/canary-ft/experiments/checkpoints/` was found to contain full, real FSDP distributed checkpoints (step 500 through 10000-last, ~5.6GB each) dated 2026-04-22/23, for a run named `canary_uwb_atcc_v3` in its own `exp_config.yaml`. This directly conflicts with the user's own prior statement ("I have lost the original training data/checkpoints/results from the earlier runs") that the entire IEEE-review-response retraining program ([[DEC-005]], [[DEC-006]], [[EXP-010]]) was built on. At the time, this surviving config's `lora.target_modules` appeared to be `["q_proj"]` only — not `["q_proj", "v_proj"]` as committed/cited. Two other undocumented historical run directories (`run_0`: LoRA r=128, `run_1`: LoRA r=64 — a rank never mentioned in any doc or the manuscript) exist as logs only, with no surviving checkpoint weights.

**CORRECTION (2026-09-08):** the `q_proj`-only claim above is **retracted**. Root cause: `exp_config.yaml` is rewritten within the first ~30–60s of *any* new training run starting. When Stage 2 was launched, it silently overwrote this same file before the emergency protective rename (done ~9 minutes into Stage 2's run, in time to save the `checkpoints/` directory but too late for `exp_config.yaml`). The file later re-read and quoted as "historical" was actually already Stage 2's own config, not the original. **Ground truth, recovered independently from the checkpoint's own embedded `meta.pt` hyperparameters** (baked in by PyTorch Lightning at save time, unaffected by any later file overwrite): `lora_dropout: 0.1`, `target_modules: ['q_proj', 'v_proj']` (both), `spec_augment` present (freq_masks=2, time_masks=10), `weight_decay: 0.01` — this **exactly matches** the documented/committed v3 recipe. See [[VAL-011]] for the full corrected finding, which is actually more serious than the original claim: a checkpoint verified to have the exact documented v3 training config still does not reproduce the manuscript's cited 20.70% WER.

Evidence (original, now superseded): `find ~/canary-ft/experiments/checkpoints -type f`; `cat ~/canary-ft/experiments/exp_config.yaml` at the time (before realizing it was contaminated). Evidence (corrected): `torch.load(.../step=10000-last.ckpt/meta.pt')['hyper_parameters']['cfg']`, read directly this session — see [[VAL-011]].

Impact: (1) The premise that all Canary-Qwen checkpoints are lost is not fully accurate — the v3 UWB-ATCC run survives and was evaluated directly. (2) The real finding is not a config mismatch but a **reproducibility gap**: correct config, wrong WER. See [[VAL-011]]. (3) No v1-equivalent baseline checkpoint survives (confirmed absent from `run_0`/`run_1`), so Stage 2 of [[EXP-010]] was correctly launched and has now completed ([[VAL-010]]).

Mitigation: The surviving checkpoint was evaluated (inference only, near-zero GPU cost) — see [[VAL-011]] for the result and its implications.

Resolution: **FULLY RESOLVED (2026-09-08) — see [[VAL-012]].** Both the config-provenance question and the WER-reproducibility question are now answered: the checkpoint's config genuinely matches the documented v3 recipe (already established here), AND its WER genuinely reproduces 20.70% — the earlier apparent non-reproduction ([[VAL-011]]) was itself caused by an unrelated caching bug in `eval_finetuned.py`, not a real problem with this checkpoint. Verified three independent ways: isolated local re-eval, independent HuggingFace download, and the author's own contemporaneous `v3_results.json` — all bit-for-bit identical. User's original assertion that this result was not fabricated is vindicated with reproducible evidence.

Related Records: [[ISS-007]], [[DEC-005]], [[DEC-006]], [[DEC-007]], [[EXP-010]], [[EXP-011]], [[VAL-011]], [[VAL-012]]

---

## ISS-006 — ATCOSIM ablation wrapper scripts lack `set -e`, silently report "Done"/"Training complete" even after all DDP ranks crash

Date: 2026-09-07
Status: RESOLVED (workaround only — no script modified, per instruction not to touch pipeline files)
Severity: HIGH (a failed run can be mistaken for a completed one from the log alone)

Description: `ablations/atcosim/train_w2v2_large-60v-{female,male}.sh` (and the shared `train_w2v2_large-60v.sh` baseline script) do not set `set -e`/`set -euo pipefail`. When the very first EXP-007 female launch attempt failed — all 4 `torchrun` DDP ranks crashed with `ModuleNotFoundError: No module named 'datasets'` (see root cause below) — the wrapper script continued past the failed `bash src/run_asr_fine_tuning.sh ...` call and printed `Done training facebook/wav2vec2-large-960h-lv60-self on ATCOSIM train_female in: experiments/results/speaker_independent_female` followed by `exit 0`. The outer wrapper `train_wav2vec2_atcosim_large_female.sh` then also printed `Training complete.` Both success messages appeared despite zero training steps having run.

Evidence: First launch attempt log (`experiments/results/speaker_independent_female/train_female.log`, first run before relaunch): `torch.distributed.elastic.multiprocessing.errors.ChildFailedError` for ranks 0-3 (`ModuleNotFoundError: No module named 'datasets'`), immediately followed in the same log by `Done training ... in: experiments/results/speaker_independent_female` and `Training complete. Results in: ...`. No checkpoint, vocab, or trainer_state file was written to the output directory (verified with `find`).

Impact: Anyone (human or agent) checking only the wrapper script's own stdout/exit message — rather than grepping for `Traceback`/`ChildFailedError`/actual step progress — would incorrectly conclude the run succeeded. This directly matches the class of failure the training-monitor completion-detection rules exist to prevent (a job is COMPLETE only on process exit + final checkpoint + final metrics, never on a printed message alone).

Root cause (of the underlying failed launch, distinct from the logging bug above): the launch was performed via a detached `setsid`/`nohup` shell (tmux was unavailable on this host — see [[ENV-004]]) that did not source `~/miniconda3/etc/profile.d/conda.sh` / `conda activate w2v2_asr`, so `torchrun` ran under the base conda env's `python3.13` (no `datasets` package) instead of the `w2v2_asr` env's `python3.10` (`datasets==2.14.0`, confirmed via `conda run -n w2v2_asr python -c "import datasets"`). A separate, unrelated `set -u`/unbound-`PYTHONPATH` failure occurred on the very first attempt for the same underlying reason (detached shell does not inherit the interactive shell's exported `PYTHONPATH`).

Mitigation: Relaunched by explicitly sourcing conda and activating `w2v2_asr` plus pre-setting `PYTHONPATH=` inside the detached `setsid nohup bash -c '...'` wrapper. No pipeline script was edited. This is a per-invocation launch fix, not a permanent one — any future detached (non-interactive) launch of these scripts must activate `w2v2_asr` and set `PYTHONPATH` explicitly, or it will silently "succeed" per this same bug.

Resolution: Retlaunch succeeded — 4 DDP rank processes confirmed alive on GPUs 0-3 via `ps`/`nvidia-smi` (100% util on 3/4 GPUs, transient 0% on the 4th consistent with dataloader activity) after the environment fix. Underlying `set -e` gap in the wrapper scripts remains unresolved in the scripts themselves (not modified, per instruction) — recommended future fix: add `set -euo pipefail` to `ablations/atcosim/train_w2v2_large-60v*.sh`.

Related Records: [[EXP-007]], [[VAL-004]], [[VAL-005]], [[ENV-004]]

---

## ISS-010 — Two live hazards found in an independent re-audit of the v1/v2/v3 rename: v1/v3 checkpoint-dir collision, and a stale destructive `rm -rf` in REPLICATION_GUIDE.md

Date: 2026-09-08
Status: OPEN (documented only, nothing modified — read-only audit per instruction)
Severity: HIGH (one is a data-loss risk to surviving checkpoints; the other blocks reproducing v1/v2 from a clean checkout without undocumented tribal knowledge)

Description: An independent, skeptical re-audit (requested explicitly, not trusting prior session summaries — see [[AUD-005]]) of the DEC-008 v1/v2/v3 rename and same-day v2 encoder-unfreeze fix found the rename/fix themselves to be correct (v1-vs-v3 regularization-only, v2-vs-v1 encoder-unfreeze-only, both verified via `diff`), but surfaced two hazards not previously recorded:

(1) **v1/v3 checkpoint collision, framework-confirmed.** `salm_uwb_atcc_v1.yaml` and `salm_uwb_atcc_v3.yaml` both set `explicit_log_dir: /home/kotasthane/canary-ft/experiments/` (identical). Reading NeMo's `exp_manager.py::check_explicit_log_dir` confirms `exp_dir`/`name`/`version` are ignored whenever `explicit_log_dir` is set — checkpoints land flatly in `<explicit_log_dir>/checkpoints/` regardless of `name`. So v1 and v3 currently target the exact same output directory; training either one after the other (without a manual backup) silently overwrites the other's checkpoints and `exp_config.yaml`. Only v2 was fixed to use an isolated directory (`experiments_v2/`); v1/v3 were not. This is the same failure mode already documented once for a different run pair in [[ISS-009]].

(2) **Wrapper-script/runtime-config desync + stale destructive doc command.** `train_canary_v1.sh` and `train_canary_v2.sh` pass `--config-path=/home/kotasthane/canary-ft/conf`, a separate non-git directory that (at audit time) does not contain `salm_uwb_atcc_v1.yaml` or `salm_uwb_atcc_v2.yaml` under those names — only the pre-rename filenames (`salm_uwb_atcc.yaml`, `salm_uwb_atcc_unfrozen.yaml`) exist there, unchanged since before DEC-008. Root cause: the rename (commit 7764ab9) renamed files in the git repo and updated the `.sh` files' `--config-name` flags, but never re-deployed the renamed files into `~/canary-ft/conf/`. `REPLICATION_GUIDE.md` §2.9/2.11/2.11b documents the required manual `cp <repo file> ~/canary-ft/conf/` step, so the wrapper scripts are not meant to run standalone — but nothing in the `.sh` files says so, and running them directly today would fail immediately (cheap Hydra "missing primary config" error, no GPU cost) for v1/v2 (v3 currently still works since its `~/canary-ft/conf/` copy happens to be current). Separately, and more seriously: `REPLICATION_GUIDE.md` §2.11 still contains `rm -rf ~/canary-ft/experiments/checkpoints/*` right before the v2 training command, written back when v2 shared v1's directory. This line was not updated when v2.yaml's `explicit_log_dir` was moved to `experiments_v2/` today — it is now pointless for its original purpose AND, per hazard (1) above, would delete both v1's and v3's live checkpoints if anyone follows the guide literally.

Evidence: `nemo/utils/exp_manager.py` (`check_explicit_log_dir`, ~line 1099-1109: "exp_dir, name, and version will be ignored"); `test -e ~/canary-ft/conf/salm_uwb_atcc_v{1,2}.yaml` → both MISSING; `md5sum` showing `~/canary-ft/conf/salm_uwb_atcc.yaml`/`salm_uwb_atcc_unfrozen.yaml` unchanged (still frozen-encoder, still identical to each other); `git show 7764ab9 -- models/canary-qwen/scripts/`; `REPLICATION_GUIDE.md` lines 165-210 (manual `cp` steps, and the `rm -rf` at line 188).

Impact: (1) risks silently destroying v1 or v3's surviving checkpoint the next time either config is retrained (e.g. for [[EXP-011]] or any future ablation re-run). (2) blocks a clean reproduction of v1/v2 via the `.sh` wrapper alone (must additionally consult `REPLICATION_GUIDE.md`); the guide's own v2 instructions currently recommend a command that would cause the exact collision described in (1) for no remaining benefit.

Mitigation: None applied — read-only audit per task instruction. Recommended fixes (not yet done): give v1 (or v3) its own `explicit_log_dir` the same way v2 was fixed; remove or correct the stale `rm -rf` line in `REPLICATION_GUIDE.md` §2.11; either add a `cp`-and-check step inside `train_canary_v1.sh`/`v2.sh` themselves or add a header comment pointing at the `REPLICATION_GUIDE.md` prerequisite so the scripts aren't silently non-functional in isolation.

Resolution: Not resolved.

Related Records: [[AUD-005]], [[ISS-007]], [[ISS-009]], [[DEC-008]]

## ISS-011 — fp16 AdamW is numerically degenerate for every Canary-Qwen run in this repo; caused S3-B3's divergence and invalidates the weight_decay attribution in v3

Date: 2026-09-09
Status: OPEN (fix implemented -- MasterWeightAdamW + StableSALM with dynamic loss scaling in models/canary-qwen/scripts/ -- Gate 1 staged validation in progress, not yet verified at production scale)
Severity: CRITICAL (invalidates a causal claim already in the research record; caused a genuine training failure)

Description: `precision: 16-true` (Lightning `HalfPrecision`) provides no GradScaler and no fp32 master weights — every parameter and every AdamW optimizer-state tensor is fp16. Verified directly on this machine:

1. **AdamW's second moment underflows to exactly zero** for realistic per-element gradients (~2.2e-5, the expected RMS gradient for a 2B-parameter model under gradient_clip_val=1.0). With `v ≡ 0`, the update denominator collapses to `eps` alone, so the optimizer silently becomes SGD-with-momentum at effective LR = `lr/eps` (5e-4/1e-4 = **5.0** for every Canary-Qwen config in this repo, including v1/v2/v3). This was survivable for LoRA (small, zero-initialized adapter subspace) but destructive for S3-B3's full-decoder fine-tune (2.0B trainable params), causing `val_loss` to diverge to `inf` by step 4000/10000.
2. **`weight_decay` has been numerically inert in every run.** Decoupled decay `p *= (1 - lr*wd)` requires `lr*wd` to exceed fp16's half-ulp-at-1.0 (4.9e-4) to have any effect. v1 (`5e-4 * 1e-3 = 5e-7`), v3 (`5e-4 * 1e-2 = 5e-6`), and the S3-B3 config all fall far below this floor — `1 - lr*wd` rounds to exactly `1.0`, verified via `torch.tensor(1 - 5e-4*1e-3, dtype=float16) == 1.0`. **This means v3's reported improvement over v1 (23.32% -> 20.70%) is attributable only to SpecAugment + LoRA dropout, not weight_decay**, despite the ablation and prior write-ups (see [[VAL-003]], EXP-004) crediting all three changes together.
3. **A second, independent bug** (FSDP2 sharding untying Qwen3's tied `embed_tokens`/`lm_head` weight, verified via `is`/`data_ptr` checks against the loaded checkpoint) meant S3-B3 was training 311M "phantom" duplicate parameters, and is the most likely concrete site of the first `inf` (an unbounded 151936-row output head trained at full LR on a tiny ATC vocabulary).

Evidence: Verified numerically this session (not inferred) — `exp_avg_sq` stays exactly 0.0 after 200 simulated steps at realistic gradient magnitudes across `eps` in {1e-4, 1e-8}; `1 - lr*wd == 1.0` for all three documented configs; `embed_tokens.weight is not lm_head.weight` (distinct `data_ptr`) post-FSDP2-sharding despite `tie_word_embeddings: true` in the base Qwen3 config; S3-B3's own training log (`val_loss` climbing 0.680->0.703->0.739->0.846 over steps 2000-3500, then `inf` at 4000/4500/5000).

Impact: (a) The training failure itself (~19 GPU-hours lost on S3-B3 before being caught and killed). (b) **Every prior WER result in this repo (v1 23.32%, v2 23.82%, v3 20.70%) was obtained under this same degenerate optimizer** — the numbers are real and reproducible (independently re-verified multiple times, see [[VAL-012]]), but the *mechanism* credited for v3's improvement is partially wrong: attribute it to SpecAugment + dropout only, not weight_decay. (c) Any future LoRA-vs-full-decoder comparison is confounded by optimizer identity unless both arms are re-run under a fixed optimizer, or the comparison is explicitly caveated as "under the documented optimizer defect."

Mitigation: Fix designed (fp32 master-weight AdamW + manual loss scaling, since `16-mixed` is blocked by `ModelParallelStrategy` and `bf16` is ~7.6x slower on this Turing-architecture hardware — measured, not assumed). Freeze `embed_tokens`/`lm_head` explicitly for any future full-decoder-scope config to close the tied-embedding bug regardless of the optimizer fix. New LR for full-decoder fine-tuning must be ~1e-5 (not 5e-4) once the optimizer is real, since the fp16 weight-resolution floor makes `lr < ~1e-5` silently do nothing and true full-FT LRs are ~50x below LoRA's. A staged Gate 0-3 validation protocol (offline numeric check -> 50 steps -> 500 steps -> 2500 steps, ~3h total) must pass before any future multi-hour run, per the new escalation rule in CLAUDE.md Section 5.

Resolution: Not yet resolved. Fix implemented (MasterWeightAdamW: fp32 master weights/moments; StableSALM: loss-scaled training_step, scaled gradient clipping, and a global-then-local on_before_optimizer_step finiteness check with dynamic loss-scale backoff/growth). Gate 0 (offline numeric check) passed. Gate 1 (50-step smoke test) surfaced and fixed two follow-on bugs before this record's initial write-up could be trusted as complete: an NCCL collective-desync hang from a per-rank-conditional DTensor collective, and a 14-20% non-finite-gradient rate under a static loss_scale (fixed via dynamic scaling + true grad=None skips instead of zero_()). Gate 1 revalidation with the dynamic-scaling fix is in progress as of this write-up. Gates 2/3 and the full 10000-step run remain outstanding.

**SUPERSEDED IN PART by [[ISS-012]] (2026-09-10)**: the "Gate 1 revalidation... in progress" line above and every conclusion drawn from Gate 1 v8 / Gate 2 v2 / Gate 2 v3 (the "14-20% -> 2-3% skip rate, dynamic loss scaling confirmed working" narrative, and the flat/worse-than-uniform val_loss=12.848 finding that motivated a bridge-layer-LR hypothesis) are INVALID as evidence about learning: checkpoint inspection proved those three runs performed exactly zero gradient-driven weight updates, due to a separate bug in gradient clipping (see ISS-012), not because the fp16-degeneracy fix here was working or not working. The optimizer/loss-scaling fix described in this record has NOT yet been validated by any run that actually updated weights. Re-validation is required after ISS-012's clipping fix lands.

Related Records: [[VAL-003]], [[VAL-009]], [[VAL-012]], [[ISS-012]], [[ISS-013]], EXP-004, EXP-012, EXP-013

## ISS-012 — Gradient clipping silently zeroed 100% of gradients in Gate 1 v8 / Gate 2 v2 / Gate 2 v3, via an fp16 norm-reduction overflow inside DTensor -- those runs performed zero weight updates

Date: 2026-09-10
Status: DIAGNOSED, fix designed, not yet implemented/verified
Severity: CRITICAL (invalidates all conclusions drawn from three staged-validation runs; was the actual, complete cause of a "flat val_loss" finding previously attributed to a different mechanism)

Description: `StableSALM.configure_gradient_clipping` (models/canary-qwen/scripts/salm_train_stable.py) calls Lightning's native `self.clip_gradients(...)`, which routes to `Precision.clip_grad_by_norm` -> `torch.nn.utils.clip_grad_norm_` operating directly on raw fp16 `p.grad` tensors -- no fp32 upcast anywhere in that path (verified by reading the installed lightning/torch source). Under FSDP2/DTensor, the per-rank local-shard norm is computed via `_NormPartial._pre_reduce_transform` (torch/distributed/tensor/_ops/_math_ops.py:129), which squares the local shard **in fp16** before the cross-rank all-reduce. fp16's max representable value is 65504, so this overflows to `inf` whenever a rank's local gradient-shard norm exceeds `sqrt(65504) ≈ 256`. Once the norm is `inf`, the clip coefficient (`max_norm / total_norm`) collapses to exactly 0, and every gradient is multiplied by zero -- silently, with no error, no NaN, and (critically) AFTER `on_before_optimizer_step`'s finiteness check already ran and passed (that check verifies the INPUT gradients are finite; clipping then destroys them afterward, invisibly to that check).

Evidence: Checkpoint-level, not inferred. `MasterWeightAdamW` stores `exp_avg`/`exp_avg_sq`/`master` in the saved DCP checkpoints for all three runs; direct inspection showed `exp_avg`/`exp_avg_sq` all exactly zero (0/2097152 nonzero for `perception.proj.weight`, 0/4194304 for a sampled decoder layer, etc.) and saved model/master weights bit-identical to the pretrained Qwen3-1.7B init (`maxabsdiff=0.000e+00`) across Gate 1 v8 (loss_scale settled at 512), Gate 2 v2 (loss_scale mostly 64, but scaled *local shard* norms still exceeded ~256 per-rank -- the earlier "safe at loss_scale=64" read used the wrong, too-permissive 65504-on-the-*global*-norm threshold), and Gate 2 v3 (loss_scale 512-1024). Cross-verified independently by two separate agents (one doing architectural analysis, one doing purely mechanical log/checkpoint verification) plus a direct empirical reproduction run by the main session (`torch.linalg.vector_norm` on a synthetic fp16 tensor at realistic magnitudes returns `inf`). All three converge on the same mechanism and the same "zero updates" conclusion.

Impact: (a) Gate 1 v8's "14-20% -> 2-3% non-finite-gradient rate, dynamic loss scaling confirmed working" conclusion in [[ISS-011]] is invalid -- that run never took a real optimizer step, so it validated nothing about the fp16-degeneracy fix. (b) Gate 2 v2's flat, worse-than-`ln(vocab_size)` val_loss (previously attributed to a randomly-initialized, undertrained `perception.proj` bridge layer trained at a mismatched LR) is now understood to be trivially explained: the model literally never changed across any of the four validation checks, since it's the same frozen pretrained-plus-random-init weights every time. The bridge-layer-LR-mismatch hypothesis is UNSUPPORTED by this observation (not disproven as a general optimization concern, but the evidence that was used to argue for it has evaporated -- see [[ISS-011]]'s superseded note). (c) [[DEC-009]]'s `GRADIENT_CLIP_VAL=1000` was calibrated from `grad_norm_unscaled` percentiles logged during these same contaminated runs; that calibration should be treated as unreliable pending a real (non-zeroed) run's gradient-norm distribution. (d) The v1/v2/v3 LoRA runs almost certainly are NOT affected by this exact bug (different numerical regime -- no artificial loss scaling, ~27.8M trainable params vs ~1.4B, smaller natural gradient magnitudes) but this has not been directly verified against those runs' own artifacts, and should not be asserted as fact without checking (per user correction, 2026-09-10: the failure condition is "a sufficiently large per-rank local shard norm reaches fp16's overflow point," which loss scaling and large parameter count make MORE LIKELY but are not strictly necessary for).

Mitigation: Fix designed (not yet implemented) -- do NOT apply the clip coefficient as an fp16-tensor multiply (risks catastrophic underflow instead: `clip_coef ~= GRADIENT_CLIP_VAL / measured_norm(200-700) ~= 1e-3`, and multiplying an fp16 gradient by an fp32 coefficient that small collapses most elements to zero on write-back to fp16). Correct design: fold the clip coefficient into the same already-correct fp32 global-norm computation `on_before_optimizer_step` already performs (reusing that single all-reduce, not a second DTensor-native one -- the replacement must NOT reintroduce the same fp16-local-norm-squaring bug via a different code path), compute `clip_coef = min(1.0, GRADIENT_CLIP_VAL / grad_norm_unscaled)` as a Python float, and apply it via `MasterWeightAdamW`'s existing fp32 divide (`grad.float() / effective_divisor`, where `effective_divisor = loss_scale / clip_coef`) rather than ever touching the fp16 gradient tensor with the coefficient directly. `configure_gradient_clipping` should become a no-op once this is in place. Additional required fixes identified during diagnosis: (1) the on_before_optimizer_step diagnostic log currently prints a garbage `grad_norm_unscaled` on skipped (non-finite) steps, since `nan_to_num(..., posinf=0.0)` silently zeroes the contribution rather than signaling invalidity -- must log NaN/a sentinel on skip steps instead; (2) the same log line prints the POST-mutation `loss_scale` next to a norm computed under the PRE-mutation scale on backoff steps, a 2x mismatch that further contaminated the percentile calibration DEC-009 used; (3) a correctness tripwire (assert some `exp_avg` is nonzero after the first successful optimizer step) should be added so this exact class of bug is caught immediately in any future run, rather than requiring a checkpoint autopsy after the fact.

Resolution: Not yet resolved. Fix design reviewed and refined via a second independent (Opus) consult; implementation pending.

Related Records: [[ISS-011]], [[DEC-009]], [[ISS-013]]

## ISS-013 — Model-provenance / evaluation-checkpoint audit required across all reported Canary-Qwen WER results (v1/v2/v3 and S3-B3): none of them fine-tune the released nvidia/canary-qwen-2.5b checkpoint, and the S3-B3 eval path may silently be evaluating the wrong model entirely

Date: 2026-09-10
Status: OPEN -- explicitly a separate review item; do NOT infer results are correct OR incorrect until the audit below is done
Severity: HIGH (could affect how every Canary-Qwen result in this project, published or in-progress, should be described; distinct from ISS-011/ISS-012's numerical-training bugs)

Description: Every Canary-Qwen config in this project (`salm_uwb_atcc_v1.yaml`, v2, v3, and the newer `salm_uwb_atcc_s3b3_fixed.yaml`) constructs its SALM model by composing `pretrained_llm: Qwen/Qwen3-1.7B` + `pretrained_asr: nvidia/canary-1b-flash` from scratch via NeMo's speechlm2 recipe (`SALM.__init__`, NOT `SALM.from_pretrained(...)`). The released `nvidia/canary-qwen-2.5b` HuggingFace checkpoint contains a TRAINED `perception.proj` (the modality-adapter projection connecting the frozen speech encoder to the Qwen3 decoder's embedding space: verified directly, `perception.proj.weight` mean=-3.35e-5, std=0.054, present in that checkpoint's 1718 keys) -- but `canary-1b-flash`'s own checkpoint (1406 keys, prefixes `encoder/log_softmax/preprocessor/transf_decoder`) has no such layer, and none of this project's training scripts ever load the full `canary-qwen-2.5b` checkpoint to populate it. This means: (a) every model this project has trained (v1/v2/v3 included, not just the new S3-B3 track) starts from a randomly-initialized modality bridge, not the released model's trained one, and (b) `PROGRESS.md`'s description of the model as "nvidia/canary-qwen-2.5b" is imprecise -- it is architecturally the same design, assembled from the same pretrained components NVIDIA used, but is a separately-initialized instance with its own training history for that one layer.

Separately, a specific and more acute risk was found in `models/canary-qwen/scripts/eval_finetuned.py`: it loads the model via `SALM.from_pretrained('nvidia/canary-qwen-2.5b')` (whose LLM is a `PeftModel` with LoRA-shaped keys, e.g. `llm.base_model.model.model.layers.*`) and then does `model.load_state_dict(state, strict=False)` with the locally-trained checkpoint. For v1/v3 (which do use LoRA) this key structure matches. For S3-B3 (no LoRA block, plain full-parameter fine-tune, keys like `llm.model.layers.*` per direct DCP metadata inspection) this is a structural mismatch -- under `strict=False`, the entire ~1.4B-parameter LLM could silently fail to load, with the returned `missing_keys`/`unexpected_keys` never inspected or asserted on, meaning `eval_finetuned.py` could report the RELEASED model's numbers for an S3-B3 run rather than the actual trained checkpoint's.

Evidence: Direct source/checkpoint inspection (not inferred) -- `salm_uwb_atcc_v1.yaml` lines 2-4 confirm the same `pretrained_llm`+`pretrained_asr` composition pattern S3-B3 uses; `nvidia/canary-qwen-2.5b`'s cached safetensors (`~/.cache/huggingface/hub/models--nvidia--canary-qwen-2.5b/.../model.safetensors`) contains `perception.proj.weight`/`.bias` with non-random statistics; `canary-1b-flash`'s cached `.nemo` archive's 1406 keys contain no `proj.*` entry at the perception namespace; `eval_finetuned.py:40-41`'s `from_pretrained` + `load_state_dict(strict=False)` pattern confirmed by direct read; S3-B3's own saved-checkpoint key prefixes (`llm.model.layers.*`, no `base_model`/`lora` wrapping) confirmed via DCP metadata, contrasted against the released checkpoint's LoRA-wrapped key prefixes.

Impact: Per user instruction (2026-09-10), explicitly NOT yet resolved in either direction -- do not conclude "v1/v2/v3's published WER numbers are unaffected" (their training regime differing from S3-B3's plausibly-broken-clipping regime is circumstantial, not a completed provenance audit) and do not conclude they need retraction either. The specific, higher-confidence risk is S3-B3's `eval_finetuned.py` path silently evaluating the released model instead of the trained one -- this is checkable directly (inspect the returned missing/unexpected key lists; compare fixed-example logits between the training-time model and the independently-reconstructed eval-time model) and should be checked before any future S3-B3 WER number is trusted.

Mitigation (audit plan, not yet executed): For every reported experimental result (v1, v2, v3, any future S3-B3 result), construct a provenance record covering: exact training initialization (component checkpoint revisions), exact trainable-module list, exact saved-artifact format (full state vs. adapter-only), exact evaluation-time initialization and loaded checkpoint, the actual `missing_keys`/`unexpected_keys` returned by the eval load (captured and asserted on, not discarded), and a fixed-batch logit-agreement check between the training-time model and the reconstructed eval-time model. `eval_finetuned.py` should be changed to capture and assert on `load_state_dict`'s return value rather than silently accepting `strict=False`'s permissiveness.

Resolution: PARTIALLY RESOLVED (2026-09-10) -- the acute eval-path risk (S3-B3-style checkpoints silently evaluating against the released model's weights) is now RULED OUT for v1 and v3 specifically, by direct evidence: ran the exact `eval_finetuned.py` load path (`SALM.from_pretrained('nvidia/canary-qwen-2.5b')` + `load_state_dict(state, strict=False)`) against both `~/canary-ft/experiments/checkpoints/step=10000-last.ckpt` (v1, [[VAL-010]]) and `~/canary-ft/experiments/checkpoints_v3_HISTORICAL_BACKUP_20260422/step=10000-last.ckpt` (v3, [[VAL-011]]/[[VAL-012]]) and captured the actual `missing_keys`/`unexpected_keys` (previously never inspected): **both returned 0 missing, 0 unexpected keys** -- a perfect structural match, confirming these two checkpoints' weights genuinely load and are what `eval_finetuned.py` actually evaluated. v1's 23.32% and v3's 20.70% are NOT artifacts of a silent fallback to the released model.

Still open: (a) the broader model-provenance/description question (v1/v2/v3 use a randomly-initialized modality bridge, not the released model's trained one -- a documentation-precision issue, not a results-validity one, already corrected in `PROGRESS.md`); (b) v2's config is separately known-unrecoverable ([[ISS-007]], unrelated to this audit); (c) the same load-check for any future completed S3-B3 checkpoint, once one exists -- `eval_finetuned.py` now asserts on this automatically (raises if missing+unexpected keys exceed 5) so this can no longer fail silently, but no real S3-B3 checkpoint has been produced yet to test it against. Per explicit user scope decision (2026-09-10): do not switch this project's model-initialization strategy (composed-from-pretrained-components) mid-repair of ISS-011/ISS-012; fine-tuning the fully-released `canary-qwen-2.5b` checkpoint instead is a legitimate but SEPARATE, not-yet-specified future experiment, answering a different question than the current "composition adaptation" track.

Related Records: [[ISS-011]], [[ISS-012]], [[VAL-010]], [[VAL-011]], [[VAL-012]], [[ISS-007]]
