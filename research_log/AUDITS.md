# Audits

## AUD-001 — ATCOSIM gender/speaker-independent split: train/test speaker overlap

Date: 2026-09-06 (recovered; underlying audit performed during Phase 4, dated in repo history around commit d11c8d8 "Add ATCOSIM gender eval results")
Status: COMPLETE (audit finding is FAIL — leakage confirmed)
Objective: Determine whether the ATCOSIM gender-based test subsets (test_female=zf3, test_male=gm1/gm2) provide a valid unseen-speaker WER measurement for the model trained on the full-corpus random 80/20 split.

Scope: ATCOSIM corpus, full-data W2V2-large model (trained on random 80/20 split across all 10 speakers), evaluated against gender-filtered subsets.

Methodology: Compared speaker membership of the gender subsets (test_female/test_male, defined by speaker ID) against the speaker composition of the random 80/20 train split.

Tools / Commands: Manual inspection of split membership as documented in models/w2v2/docs/PROGRESS_ATCOSIM.md "Phase 4 - Gender Experiments" section (no separate script found in repo).

Findings:
- The random 80/20 split used for the main ATCOSIM training run drew utterances from all 10 speakers into both train and test.
- The gender subsets (test_female = zf3; test_male = gm1, gm2) are simply the full corpus filtered by speaker, not a re-split.
- ~80% of each of zf3's, gm1's, and gm2's utterances were already present in the training set used to produce the evaluated checkpoint.
- Reported numbers under this invalid setup: test_female (zf3): 86% WER (eval_model.py) / 0.86% WER (custom script); test_male (gm1, gm2): 1.23% WER (eval_model.py) / 0.01% WER (custom script).
- A second, independent bug was found in eval_model.py: when no LM is provided it substitutes the CTC-decoded label as the "reference" text in the printed hypothesis file, inflating/deflating the reported WER relative to the custom script (see ISS-002).

Evidence: models/w2v2/docs/PROGRESS_ATCOSIM.md, section "Phase 4 - Gender Experiments (Speaker-Independent Eval) [ATTEMPTED — DATA LEAKAGE FOUND]"; shared/data_info.md gender-subset speaker table.

Conclusion: The gender-subset WER numbers (0.86%, 0.01%, 86%, 1.23%) are NOT valid speaker-independent evaluations and must not be reported as ATCOSIM generalization results. They are discarded findings, kept on record for transparency, not for reuse.

Limitations: The audit was manual (documentation-based); no automated overlap script/output was found in the repo. Utterance-level overlap percentage ("~80%") is stated in prose, not backed by a machine-produced count file in this repo.

Related Records: [[ISS-001]], [[ISS-002]], [[DEC-001]]

---

## AUD-002 — UWB-ATCC / ATCOSIM data split and format audit (as documented)

Date: 2026-09-06 (recovered; underlying data prep performed in Phase 2 of each corpus)
Status: COMPLETE
Objective: Record the verified train/test split sizes and formats for both corpora, as produced by the repo's data preparation scripts.

Scope: UWB-ATCC and ATCOSIM corpora, Kaldi-format outputs used by the W2V2 pipeline, and NeMo JSONL manifests used by the Canary pipeline.

Methodology: Read documented output of `scripts/data_prepare_uwb_atcc.sh` and `scripts/data_prepare_atcosim.sh` as recorded in PROGRESS docs and shared/data_info.md.

Tools / Commands: N/A (values recorded from script run logs in the docs; scripts themselves are in models/w2v2/scripts/).

Findings:
- UWB-ATCC (80/20 split, seed=1234): Train = 11,543 utterances / 2,086 recordings (~10.5 hrs); Test = 2,886 utterances / 570 recordings (~2.6 hrs). Total corpus duration 20.58h, 8kHz native, resampled to 16kHz for Canary.
- ATCOSIM (80/20 split, seed=1234): Train = 7,660 utterances; Test = 1,916 utterances. Total corpus ~10h, 32kHz native, resampled to 16kHz.
- ATCOSIM speakers: 10 total — 6 male (gm1, gm2, sm1, sm2, sm3, sm4), 4 female (gf1, zf1, zf2, zf3).
- ATCOSIM gender subsets: train_female = {zf1, zf2, gf1}; test_female = {zf3}; train_male = {sm1, sm2, sm3, sm4}; test_male = {gm1, gm2}. These subsets are speaker-filtered views of the same corpus, not an independent re-split (see AUD-001, ISS-001).
- ATCOSIM min_duration_in_seconds had to be raised from paper's 0.2 to 0.5 — clips shorter than this crashed the SpecAugment time-mask (seq_len < mask_len=12) (models/w2v2/docs/PROGRESS_ATCOSIM.md Phase 2 hyperparameter table).

Evidence: shared/data_info.md; models/w2v2/docs/PROGRESS_UWB_ATCC.md Phase 2; models/w2v2/docs/PROGRESS_ATCOSIM.md Phase 1-2.

Conclusion: Both corpora's stated split sizes are consistent across README.md, SUMMARY.md, shared/data_info.md, and the per-corpus PROGRESS docs — no discrepancy found among these files.

Limitations: No raw manifest/utt2spk files are present in this repo (data lives outside the repo per REPLICATION_GUIDE.md paths like `~/w2v2-air-traffic/experiments/data/...`), so split sizes could not be independently re-verified from raw files in this pass — they are taken as consistent, cross-referenced documentation, not re-derived from data.

Related Records: [[AUD-001]], [[ENV-002]]

---

## AUD-003 — Documentation self-consistency corrections (WER metric naming, encoder-unfrozen param %)

Date: 2026-09-06 (recovered; corrections made 2026-06-11 and 2026-06-12 per commit dates)
Status: COMPLETE
Objective: Record two verified documentation-correction passes found in git history, since they change previously reported numbers/labels that could otherwise be mistaken for two different results.

Scope: SUMMARY.md, PROGRESS_UWB_ATCC.md, PROGRESS_ATCOSIM.md, REPLICATION_GUIDE.md, models/canary-qwen/docs/PROGRESS.md, shared/model_comparison.md.

Methodology: `git show` on commits 34a5ca1 and 054bd54.

Tools / Commands: `git show --stat 34a5ca1`, `git show 34a5ca1`, `git show --stat 054bd54`, `git show 054bd54`

Findings:
1. Commit 34a5ca1 (2026-06-11): The UWB-ATCC no-LM decoding method was mislabeled "beam search" throughout docs; corrected to "greedy" (eval_model.py uses greedy decoding when no LM is supplied). Alongside this, the reported average WER of 14.57%/12.76% (an average across 2 eval runs: 14.54%/12.69% and 14.60%/12.82%) was replaced by the single canonical run stored in `finetuned_results_v2.json`: 14.54% (no LM) / 12.69% (with KenLM). The 14.60%/12.82% run-to-run variation is preserved as a note, not discarded.
2. Commit 054bd54 (2026-06-12): The Canary-Qwen "encoder unfrozen" trainable-parameter percentage was corrected from 32.8% to 29.2% (838.8M / 2,870M), and the "33x more parameters" comparison text corrected to "30x", across REPLICATION_GUIDE.md, models/canary-qwen/docs/PROGRESS.md, and shared/model_comparison.md.

Evidence: `git show 34a5ca1`, `git show 054bd54` (see diffs above).

Conclusion: The current canonical numbers in the repo are: UWB-ATCC W2V2 = 14.54% (no LM, greedy) / 12.69% (CTC+KenLM); Canary-Qwen encoder-unfrozen = 838.8M / 2,870M = 29.2% trainable.

Limitations: `models/canary-qwen/docs/finetuned_results_unfrozen.json` still stores the old value `"params_trained_pct": 32.8` — the JSON artifact was not updated when the docs were corrected. This is a stale-data discrepancy, not a re-derivation; flagged as [[ISS-004]]. Per user instruction, this audit does NOT modify that file or any existing research documentation.

Related Records: [[ISS-004]], [[DEC-002]]

---

## AUD-004 — Full repository structure/scripts/docs audit and hardware/software efficiency review

Date: 2026-09-07
Status: COMPLETE

Objective: User-requested audit of the entire `Pilot-to-ATC-Research` repo (GitHub structure, all scripts, all `.md` docs) plus a hardware/software training-efficiency review, "to help prepare for the research."

Scope: Full repo tree (`find`), all `.sh` scripts in both `Pilot-to-ATC-Research` and the cloned `w2v2-air-traffic`, all `.md` docs (README.md, SUMMARY.md, REPLICATION_GUIDE.md, `models/{w2v2,canary-qwen}/docs/*.md`, `shared/*.md`), and every training config's GPU/batch allocation.

Methodology: Direct `find`/`grep`/`cat`/`Read` inspection; no agents. Cross-checked every WER/param/step figure across all docs against each other and against the underlying JSON result files.

Findings:
1. **GitHub structure**: `.gitignore` correctly excludes model weights/audio/logs — no bloat. README.md's "Repository Structure" diagram is stale — missing `train_wav2vec2_atcosim_large_{female,male}.sh` (added 2026-09-06) and `research_log/` entirely.
2. **Scripts — two incompatible environment conventions coexist, undocumented**: UWB-ATCC-generation scripts (`train_wav2vec2_large.sh`, `train_wav2vec2_base_*.sh`, `eval_large_model.sh`, `train_kenlm.sh`) have `conda activate w2v2_asr` but no `set -e`. ATCOSIM-generation scripts (`train_wav2vec2_atcosim_large.sh` baseline + `_female`/`_male`, `data_prepare_*.sh`) have `set -euo pipefail` but **no `conda activate`** — this exact gap caused the real launch failure documented in [[ISS-006]]/[[ENV-004]], and it predates the female/male scripts (the ATCOSIM baseline itself has the same gap).
3. **Stale-number bug survived the original fix, in a NEW location**: `models/canary-qwen/scripts/train_canary_unfrozen.sh`'s header comment still reads "32.8% params" — commit 054bd54 (see AUD-003 above) fixed this everywhere except this script and `finetuned_results_unfrozen.json` (already known, [[ISS-004]]).
4. **Documented prerequisite doesn't exist**: `train_wav2vec2_large.sh` says "See docs/SETUP.md for sed commands" to switch `python3`→`torchrun`; `SETUP.md`'s 8 "Known Issues" do not include this fix anywhere. A new user following `REPLICATION_GUIDE.md` from scratch has no documented step for the DDP switch that every training script assumes is already done.
5. **Docs don't reflect ATCOSIM Phase 4 completion (now resolved by EXP-007, this session)**: at the time of this audit, `SUMMARY.md`'s "Pending/Next Steps" and `PROGRESS_ATCOSIM.md`'s Phase 4 header both still said the speaker-independent ATCOSIM re-run hadn't happened. `SUMMARY.md` also mischaracterized the design as cross-gender train/test, contradicting `PROGRESS_ATCOSIM.md`'s own (correct) within-gender description.
6. **Hardware/software efficiency**: current large-model configs (W2V2 batch=1/DDP, Canary-Qwen FSDP) are correctly minimal for the 11GB-VRAM constraint. Two historical exploratory runs were flagged as resource-inefficient: (a) the UWB-ATCC W2V2-base 3,000-step "pipeline validation only" run used all 4 GPUs when 1 would suffice; (b) the Canary-Qwen "Lower LR" and "Research-optimized" ablations both ran to their full fixed step count despite trajectories that were likely visibly bad well before the final step — a staged/gated step-count protocol would have saved GPU-hours without losing information.

Evidence: direct file reads and greps as described in Methodology; specific line/file references given in the chat report delivered to the user in this session (not reproduced verbatim here to keep this record concise — see the full text if needed by searching this session's transcript, or re-run the same `find`/`grep` commands, all of which are reproducible from the commands listed).

Conclusion: No numeric drift found in any cross-checked WER/param/step figure across README/SUMMARY/REPLICATION_GUIDE/PROGRESS_ATCOSIM/model_comparison.md (all consistent post-054bd54) — the issues found are structural (script conventions, doc staleness, one new stale-number instance, one documentation gap), not new instances of numeric drift.

Limitations: This audit was performed via chat report at the time and **was not written to this file until 2026-09-07 (same day), after the omission was caught during a later engineering-memory update** — flagging this explicitly per the project's own "never claim historical knowledge that was not actually retrieved or verified" policy. No files were modified during the original audit; this entry is a faithful reconstruction of that audit's actual findings, not a re-derivation.

Related Records: [[ISS-004]], [[ISS-006]], [[ENV-004]], [[EXP-007]], [[AUD-003]]

---

## AUD-005 — Independent fresh-eyes re-audit of Canary-Qwen UWB-ATCC v1/v2/v3 config-naming fix (post-DEC-008/ISS-007)

Date: 2026-09-08
Status: COMPLETE

Objective: Independently re-verify, from the actual files (not from prior session summaries), that the v1/v2/v3 rename (commit 7764ab9, DEC-008) and the same-day v2.yaml encoder-unfreeze fix (uncommitted at audit time) are correct and unambiguous — requested explicitly as a skeptical, non-trusting second check after repeated config-confusion incidents this session (ISS-007, ISS-009).

Scope: `models/canary-qwen/scripts/{salm_uwb_atcc_v1,v2,v3}.yaml`, `train_canary_{v1,v2,v3}.sh`, plus the three unrelated local-only `~/canary-ft/conf/salm_uwb_atcc_{lr1e4,optimized,optimized_v2}.yaml` ablations. Read-only; no files modified.

Methodology: Read all files directly; `diff`/`md5sum` cross-checks between repo copies and the separate `~/canary-ft/conf/` runtime directory; `git log`/`git show` on the rename commit to find root cause of a discrepancy; read `nemo/utils/exp_manager.py` (`check_explicit_log_dir`) to confirm actual `explicit_log_dir` vs `name` isolation behavior at the framework level (not just inferred from comments); grepped `REPLICATION_GUIDE.md` and repo docs for stale references.

Findings:
1. **CONFIRMED CORRECT**: v1 vs v3 differ only in regularization (`lora_dropout` 0.01→0.1, `spec_augment` block added, `weight_decay` 1e-3→1e-2). `freeze_params`, LoRA `r`/`alpha`/`target_modules`, optimizer `lr`/`betas`/`eps`, trainer block, data block all byte-identical. Verified via full `diff`.
2. **CONFIRMED CORRECT**: v2 (current, uncommitted state) vs v1 differ only in one functional respect — `^perception\.encoder\..+$` removed from `freeze_params` (genuine encoder unfreeze) — plus one *necessary, deliberate* infra change (`explicit_log_dir` moved to a separate `~/canary-ft/experiments_v2/`, `name` changed) with a clear header comment explaining why. LoRA block and optimizer identical to v1. Verified via full `diff`.
3. **NEW FINDING, framework-verified (not previously documented anywhere in this repo's research_log)**: v1 and v3 both still set `explicit_log_dir: /home/kotasthane/canary-ft/experiments/` (identical path). Read NeMo's `exp_manager.py::check_explicit_log_dir` directly — confirms that when `explicit_log_dir` is set, `exp_dir`/`name`/`version` are explicitly ignored and checkpoints land flatly in `<explicit_log_dir>/checkpoints/` regardless of `name`. This means **v1 and v3 currently collide on output directory** — training v3 after v1 (or vice versa) without a manual backup will overwrite the other's checkpoints/`exp_config.yaml`. Only v2 was fixed to avoid this; v1/v3 were not. This is the exact failure mode already documented as having happened once for a different pair of runs (ISS-009, "v1's launch silently overwrote a historical exp_config.yaml").
4. **NEW FINDING**: `train_canary_v1.sh`/`train_canary_v2.sh` pass `--config-path=/home/kotasthane/canary-ft/conf` (a separate, non-git-tracked directory), and at audit time that directory does **not** contain `salm_uwb_atcc_v1.yaml` or `salm_uwb_atcc_v2.yaml` (verified via `test -e`; only the pre-rename names `salm_uwb_atcc.yaml`/`salm_uwb_atcc_unfrozen.yaml` exist there, both still MD5-identical to each other/frozen-encoder). Root cause (via `git show 7764ab9`): the DEC-008 rename changed filenames inside the git repo and updated the wrapper scripts' `--config-name` flags to match, but never re-deployed the renamed files into the external `~/canary-ft/conf/` runtime directory that Hydra actually reads at `--config-path`. `REPLICATION_GUIDE.md` §2.9/2.11/2.11b documents a required manual `cp <repo file> ~/canary-ft/conf/` step before each wrapper script is run — so the wrapper scripts are not meant to be run standalone, but nothing in the `.sh` files themselves says so. If `train_canary_v1.sh`/`v2.sh` were invoked directly right now (skipping the guide's manual step), Hydra would fail immediately with a missing-primary-config error before any GPU/torchrun work starts (cheap failure, not a wasted-compute risk, but a correctness/reproducibility trap). `train_canary_v3.sh` is unaffected because its `~/canary-ft/conf/` copy happens to still be current (verified byte-identical via `diff`).
5. **NEW FINDING**: `REPLICATION_GUIDE.md` §2.11 (v2 training instructions) still contains `rm -rf ~/canary-ft/experiments/checkpoints/*` immediately before the v2 training command — written under the pre-fix v2 config that shared v1's `explicit_log_dir`. This line was not updated alongside today's v2.yaml `explicit_log_dir` change (to `experiments_v2/`), so it is now stale in two ways: (a) pointless for its original purpose (isolating v2), since v2 no longer writes there; (b) still destructive — because of finding 3, that shared directory currently holds both v1's and v3's checkpoints, so literally following this guide today would delete both when "preparing" for a v2 run that no longer even needs it.
6. **CONFIRMED**: the three `~/canary-ft/conf/`-only ablation configs (`salm_uwb_atcc_lr1e4.yaml`, `salm_uwb_atcc_optimized.yaml`, `salm_uwb_atcc_optimized_v2.yaml`) are not part of the canonical v1/v2/v3 set and are not referenced as if they were anywhere in repo docs (checked via `rg`). Confirmed the "_v2" in `salm_uwb_atcc_optimized_v2.yaml` is an unrelated, coincidental "v2" (LoRA r=128/eps=1e-4 vs `optimized.yaml`'s r=64/eps=1e-6 — nothing to do with encoder freezing; encoder is frozen in *both* optimized variants). Also noted: `optimized.yaml` and `optimized_v2.yaml` both internally use the *same* `name: canary_uwb_atcc_optimized` (the "_v2" exists only in the filename, not inside either config) and both also point at the same shared `~/canary-ft/experiments/` directory as v1/v3 — a latent collision if either dropped ablation were ever resurrected, though currently moot per DEC-007.

Evidence: `diff -u salm_uwb_atcc_v1.yaml salm_uwb_atcc_v3.yaml`, `diff -u salm_uwb_atcc_v1.yaml salm_uwb_atcc_v2.yaml`, `md5sum` across all 9 files (repo + `~/canary-ft/conf/`), `git show 7764ab9 -- models/canary-qwen/scripts/`, `nemo/utils/exp_manager.py` lines ~1099-1109, `rg` over `REPLICATION_GUIDE.md`.

Conclusion: The v1/v3 regularization-only and v2 encoder-unfreeze-only diffs are both genuinely correct and match the manuscript's stated design (task items 3-4 of this audit: PASS). However, two live, previously-unflagged hazards remain: (a) v1/v3 checkpoint-directory collision, framework-confirmed, not just comment-asserted; (b) the wrapper-script/`~/canary-ft/conf/` deployment gap plus a stale destructive `rm -rf` in `REPLICATION_GUIDE.md` §2.11 that, if run today, would delete both v1's and v3's surviving checkpoints for no remaining benefit.

Limitations: Did not execute the wrapper scripts or Hydra config resolution live (would require either GPU launch or a separate cheap Hydra-compose smoke test outside this audit's read-only scope) — the missing-file conclusion is based on direct filesystem inspection (`test -e`, `ls -la`) plus reading `hydra_runner`'s single-directory resolution contract in `salm_train.py`, which is standard, well-documented Hydra behavior, not empirically re-executed here.

Related Records: [[ISS-007]], [[ISS-009]], [[ISS-010]], [[DEC-008]], [[VAL-012]]
