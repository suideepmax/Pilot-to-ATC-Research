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
Status: OPEN (flagged only, not modified — instructed not to alter existing research documentation/results)
Severity: LOW

Description: `models/canary-qwen/docs/finetuned_results_unfrozen.json` stores `"params_trained_pct": 32.8`, the pre-correction value. Commit 054bd54 corrected the same figure (838.8M / 2,870M = 29.2%) everywhere in Markdown docs (REPLICATION_GUIDE.md, models/canary-qwen/docs/PROGRESS.md, shared/model_comparison.md) but did not touch this JSON file.

Evidence: `cat models/canary-qwen/docs/finetuned_results_unfrozen.json` → `"params_trained_pct": 32.8`; contrast with [[AUD-003]] / commit 054bd54.

Impact: Any future script or reader that consumes the JSON file directly (rather than the Markdown docs) will pick up the superseded 32.8% figure.

Mitigation: None applied — recorded here per instruction not to modify existing files during this audit.

Resolution: Not resolved. Recommended next action: update the JSON's `params_trained_pct` to 29.2 in a future, explicitly-scoped edit.

Related Records: [[AUD-003]]

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
