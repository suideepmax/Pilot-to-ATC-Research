# Validation

## VAL-001 — UWB-ATCC W2V2-large final checkpoint beats paper baseline

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Verify whether the locally fine-tuned W2V2-large checkpoint beats the paper's (Zuluaga-Gomez et al.) published UWB-ATCC results.

Inputs / Configuration: Final checkpoint from [[EXP-001]] (Run 3, or equivalent final checkpoint used for Phase 5 eval); 4-gram KenLM trained on UWB-ATCC train transcripts (`experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary`).

Procedure: `bash models/w2v2/scripts/eval_large_model.sh` (paper's eval_model.py-based pipeline), run standalone twice against the same checkpoint.

Expected Result: Beat or match paper's Table 3 numbers (17.48% no-LM / 14.26% with LM) and HuggingFace card numbers (17.56% / 13.72%).

Actual Result: Run A: 14.54% (no LM, greedy) / 12.69% (CTC+KenLM). Run B (repeat standalone eval, same checkpoint): 14.60% / 12.82%. Canonical stored result (`finetuned_results_v2.json`): 14.54% / 12.69%.

Evidence: models/w2v2/docs/PROGRESS_UWB_ATCC.md Phase 5; `git show 34a5ca1` (correction commit establishing canonical numbers and greedy/beam-search labeling — see [[DEC-002]]).

Result: PASS — beats paper on both no-LM and with-LM metrics, across both standalone eval runs.

Remaining Risks: Run-to-run variation of ~0.06-0.13pp observed between two standalone evals of the same checkpoint; source of variation (nondeterminism in decoding/data loading) not further investigated in-repo.

Related Records: [[EXP-001]], [[DEC-002]]

---

## VAL-002 — ATCOSIM W2V2-large final checkpoint matches paper's 20k-step result in 5k steps

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Verify whether the 5,000-step ATCOSIM W2V2-large run matches the paper's 20,000-step published result.

Inputs / Configuration: Final checkpoint from [[EXP-002]]; 4-gram KenLM trained on ATCOSIM train transcripts (`experiments/data/atcosim_corpus/train/lm/atcosim_corpus_4g.binary`).

Procedure: `python3 src/eval_model.py --pretrained-model $MODEL --test-set experiments/data/atcosim_corpus/test --print-output true`, with and without the KenLM.

Expected Result: Approach the paper's HuggingFace-card 20,000-step number of 1.67% WER.

Actual Result: 1.67% WER (no LM, greedy) — exact match to paper's 20k-step figure, achieved in 5,000 steps (1/4 the steps). With KenLM: 1.28%.

Evidence: models/w2v2/docs/PROGRESS_ATCOSIM.md Phase 2 ("Eval (paper's eval_model.py, no LM)") and Phase 3 (KenLM results).

Result: PASS, with an important caveat — see Remaining Risks.

Remaining Risks: This result is confounded by train/test speaker overlap (random 80/20 split spans all 10 speakers) and heavy training (~42 epochs on a small corpus), per [[ISS-001]]/[[DEC-001]]. The 1.67%/1.28% figures are valid as "matches paper's same-protocol number" but are NOT validated as a speaker-independent generalization result. A speaker-independent re-validation is proposed but not executed ([[EXP-007]]).

Related Records: [[EXP-002]], [[ISS-001]], [[DEC-001]]

---

## VAL-003 — Canary-Qwen v3 regularization breaks the ~24% WER plateau on UWB-ATCC

Date: recovered 2026-09-06
Status: COMPLETE

Objective: Verify whether adding SpecAugment + increased dropout + increased weight decay (v3 config) actually improves over the original LoRA config (Run 1) and the encoder-unfrozen config (Run 2), rather than the plateau being an architectural ceiling.

Inputs / Configuration: Run 1 (original LoRA, no SpecAugment, dropout=0.01, WD=1e-3) vs. Run 6/v3 (same LR, SpecAugment ON, dropout=0.1, WD=1e-2) — see [[EXP-004]] for full configs.

Procedure: Train both configs for 10,000 steps under identical LR/warmup/GPU setup; compare learning curves at matched steps (500-sample eval subset) and full-test-set final WER.

Expected Result: If the plateau is architectural (frozen LLM decoder is the bottleneck), added regularization should not meaningfully change the ~24% ceiling. If it is overfitting, regularization should lower WER.

Actual Result: v3 WER (full test set) = 20.70% vs. Run 1's 23.32% — a 2.62pp absolute improvement. At every learning-curve checkpoint from step 500 onward, v3's 500-sample WER is at or below Run 1's, and v3 keeps improving through step 10,000 (22.30%) while Run 1 plateaus (24.53% from step 7,500 onward, per `learning_curve.json`/`learning_curve_v3.json`).

Evidence: models/canary-qwen/docs/learning_curve.json, learning_curve_v3.json, finetuned_results_v2.json, v3_results.json; models/canary-qwen/docs/PROGRESS.md "Key Findings" item 1.

Result: PASS — confirms the plateau was overfitting, not an architectural limit; regularization is the load-bearing change, not the frozen decoder.

Remaining Risks: The frozen Qwen3-1.7B decoder is still cited (shared/model_comparison.md observation 6) as a likely contributor to the remaining ~6pp gap vs. W2V2 (20.70% vs 14.54%) — this is an INTERPRETATION, not independently isolated/verified by an ablation that unfreezes the decoder itself (only the encoder was tried unfrozen, in Run 2, which did not include regularization).

Related Records: [[EXP-004]], [[AUD-003]]

---

## VAL-004 — EXP-007 female/male scripts confirmed to use torchrun/DDP identically to baseline

Date: 2026-09-07
Status: PASS

Objective: Resolve GPU execution ambiguity for `models/w2v2/scripts/train_wav2vec2_atcosim_large_{female,male}.sh` before production launch — confirm process model, actual GPU usage, and equivalence to the baseline's execution mechanism ([[DEC-003]]).

Inputs / Configuration: `train_wav2vec2_atcosim_large_female.sh` / `_male.sh` → `CUDA_VISIBLE_DEVICES=0,1,2,3 bash ablations/atcosim/train_w2v2_large-60v-{female,male}.sh` → `src/run_asr_fine_tuning.sh`.

Procedure: Read-only inspection of the full call chain (wrapper scripts → `run_asr_fine_tuning.sh` → `run_speech_recognition_ctc.py`) and comparison against the baseline `train_w2v2_large-60v.sh`. No training was executed.

Expected Result: Per DEC-003, the pipeline should launch via `torchrun`/DDP (one process per GPU), not `DataParallel`.

Actual Result: `src/run_asr_fine_tuning.sh:104` runs `torchrun --nproc_per_node=4 src/run_speech_recognition_ctc.py` — a hardcoded literal, not parameterized. `run_speech_recognition_ctc.py:455-461` logs `training_args.local_rank`, `.device`, `.n_gpu`, `distributed training: {local_rank != -1}` sourced from HF `TrainingArguments`, confirming the standard DDP code path (never DataParallel) whenever launched with `WORLD_SIZE>1`. `CUDA_VISIBLE_DEVICES=0,1,2,3` restricts visible devices to 4; torchrun spawns exactly 4 ranks, each pinned via `LOCAL_RANK` to one visible device — so all 4 GPUs are used. Effective batch size = `per_device_train_batch_size(1) × gradient_acc(16) × world_size(4) = 64`, identical to the baseline's effective batch and to the math documented in the female/male scripts' own comments (verified against `ablations/atcosim/train_w2v2_large-60v.sh`, which uses the same `per_device_train_batch_size=1`/`gradient_acc=16`).

Evidence: `src/run_asr_fine_tuning.sh:104`; `src/run_speech_recognition_ctc.py:455-461`; `ablations/atcosim/train_w2v2_large-60v.sh` (baseline); `ablations/atcosim/train_w2v2_large-60v-{female,male}.sh` (comments documenting the effective-batch-64 scaling).

Result: PASS — female/male launch mechanism is execution-identical to the approved baseline; no DataParallel/DDP ambiguity remains for EXP-007.

Remaining Risks: `--nproc_per_node=4` is hardcoded with no CLI override, so a true reduced-GPU-count smoke test is not currently possible without editing `run_asr_fine_tuning.sh`.

Related Records: [[DEC-003]], [[EXP-007]], [[VAL-005]]

---

## VAL-005 — EXP-007 final preflight validation (manifests, config, output isolation, resource plan)

Date: 2026-09-07
Status: PASS

Objective: Final go/no-go preflight for EXP-007 (speaker-independent ATCOSIM re-training) before requesting production-launch approval.

Inputs / Configuration: `experiments/data/atcosim_corpus/{train_female,test_female,train_male,test_male}` manifests; `models/w2v2/scripts/train_wav2vec2_atcosim_large_{female,male}.sh`; `ablations/atcosim/train_w2v2_large-60v-{female,male}.sh`.

Procedure: Deterministic inspection only (`cut`/`comm`/`wc`/`du`/`df`/`python3 -c json`), no training executed. Checked: (a) speaker-set disjointness train vs test per gender; (b) recording/utterance-id overlap train vs test per gender and cross-gender; (c) manifest line-count consistency across `wav.scp`/`segments`/`text`/`utt2spk`; (d) sample-rate pipeline in `wav.scp`; (e) hyperparameters vs. baseline; (f) `output_dir` collision risk; (g) disk usage/free space; (h) runtime/GPU-hour estimate from the baseline's actual `trainer_state.json`.

Expected Result: Zero train/test speaker or utterance overlap; config matching baseline; no output collision.

Actual Result:
- Speaker sets: train_female={gf1,zf1,zf2} vs test_female={zf3} — disjoint. train_male={sm1-4} vs test_male={gm1,gm2} — disjoint.
- Recording overlap (train vs test, per gender): 0. Utterance-id overlap (train vs test, per gender): 0. Cross-gender (train_female vs train_male) utterance-id overlap: 0.
- Manifest counts match exactly: train_female 3,471 / test_female 616 / train_male 4,849 / test_male 640 utterances, identical across `wav.scp`, `segments`, `text`, `utt2spk`.
- `wav.scp` confirms 32kHz source resampled to 16kHz via `sox ... -r16k` on the fly.
- Hyperparameters (model, `per_device_train_batch_size=1`, `gradient_acc=16`, `learning_rate=5e-4`, `mask_time_prob=0.01`) identical to baseline; only dataset paths, `max_steps` (2500/3000, scaled to baseline's ~42-epoch exposure), and `--exp` output root differ.
- `output_dir` formula (`$exp/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_..._16acc/`) verified against the baseline's actual on-disk path; `$exp` root differs per run (`speaker_independent_female`/`speaker_independent_male` vs `baselines`) so no collision is possible. Neither output directory exists yet (clean).
- Baseline checkpoint size = 9.4GB (`save_total_limit=1`); free disk = 1.3TB — no concern.
- Baseline `trainer_state.json`: `train_runtime=13686.7s` for 5000 steps (0.365 steps/sec) on this same 4-GPU DDP config → female (2500 steps) ≈ 1.9h, male (3000 steps) ≈ 2.3h, sequential total ≈ 4.2h wall-clock ≈ 16.7 GPU-hours combined (4 GPUs × wall-clock, both runs use all 4 GPUs so must run sequentially, not concurrently).

Evidence: raw `comm`/`cut`/`wc` output on the manifests above; `wav.scp` sample line; baseline `trainer_state.json` at `experiments/results/baselines/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc/trainer_state.json`; `du -sh`/`df -h` output.

Result: PASS — no scientific, data, or output-safety blocker for EXP-007 production launch.

Remaining Risks: See [[ISS-005]] (existing KenLM 4-gram binary trained on leaked data — must not be used for this experiment's decoding). No smoke test has actually been executed for either the male or female run in this repository as of this record — only static config/manifest inspection. A smoke test (per the ML training gate) is still outstanding before production launch, or the user may choose to accept the equivalence-to-baseline evidence in lieu of one.

Related Records: [[VAL-004]], [[EXP-007]], [[ISS-001]], [[DEC-003]], [[ISS-005]]

---

## VAL-006 — EXP-007 female run completion verification (real evidence, not wrapper message)

Date: 2026-09-07
Status: PASS

Objective: Verify the EXP-007 female training run actually completed successfully before starting the male run, per [[ISS-006]] (wrapper scripts print "Done training"/"Training complete" even after a crash) and the completion-detection rules in the training-monitor skill.

Procedure: Deterministic checks against the actual log and output directory, not the wrapper's printed message alone: (a) `pgrep` for any surviving training process; (b) full-log grep for `Traceback`/`CUDA out of memory`/`ChildFailedError`/`FAILED`; (c) `grep -c "Process rank:"` to confirm all 4 DDP ranks logged in (not a silent fallback to fewer); (d) final `epoch`/step count in the log; (e) presence of a checkpoint directory and `eval_results.json`/`train_results.json`/`trainer_state.json` in the output directory; (f) reading the actual metric values from `eval_results.json`.

Expected Result: Clean exit, 4 ranks, `max_steps=2500` reached, checkpoint + eval artifacts present, no failure signatures.

Actual Result: No training process remained running (clean exit). No failure signatures anywhere in the full log. `grep -c "Process rank:"` = 4 (all DDP ranks participated). Final epoch = 47.17 (consistent with `max_steps=2500` on 3,471 train utterances / effective batch 64). `checkpoint-2000/` present plus final top-level model files (`pytorch_model.bin`, `config.json`, etc.) and `eval_results.json` (`eval_wer=0.048468446303358344`, `eval_loss=0.08569025993347168`), `train_results.json` (`train_loss=0.3924719829559326`, `train_runtime=7006.3094s`), `trainer_state.json` — all present and mutually consistent with the log.

Evidence: `experiments/results/speaker_independent_female/train_female.log`; `experiments/results/speaker_independent_female/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc/{eval_results.json,train_results.json,trainer_state.json,checkpoint-2000/}`.

Result: PASS — female run genuinely completed. WER = 4.8468% on held-out `test_female` (zf3), replacing ISS-001's leaked 0.86% figure with a legitimate speaker-independent measurement.

Remaining Risks: None for this run specifically. General risk from [[ISS-005]] still applies if a future step adds LM fusion to decoding.

Related Records: [[EXP-007]], [[VAL-005]], [[ISS-006]], [[ISS-001]]

---

## VAL-007 — EXP-007 male run completion verification (real evidence, not wrapper message)

Date: 2026-09-07
Status: PASS

Objective: Same as [[VAL-006]], applied to the male run.

Procedure: Identical deterministic checklist as VAL-006: `pgrep` for surviving processes, full-log failure grep, DDP rank count, final epoch/step, checkpoint + eval/train artifact presence, and reading the actual metric values.

Actual Result: No training process remained running. No failure signatures in the full log. `grep -c "Process rank:"` = 4. Final epoch = 39.99 (consistent with `max_steps=3000` on 4,849 train utterances / effective batch 64). `checkpoint-3000/` present plus final top-level model files and `eval_results.json` (`eval_wer=0.19973009446693657`, `eval_loss=0.5306041240692139`), `train_results.json` (`train_loss=0.4119242922465007`, `train_runtime=8623.6473s` ≈ 2.40h), `trainer_state.json` — all present and consistent with the log.

Evidence: `experiments/results/speaker_independent_male/train_male.log`; `experiments/results/speaker_independent_male/wav2vec2-large-960h-lv60-self/atcosim_corpus/0.0ld_0.0ad_0.0attd_0.0fpd_0.01mtp_12mtl_0.0mfp_12mfl_16acc/{eval_results.json,train_results.json,trainer_state.json,checkpoint-3000/}`.

Result: PASS — male run genuinely completed. WER = 19.973% on held-out `test_male` (gm1, gm2), replacing ISS-001's leaked 0.01% figure with a legitimate speaker-independent measurement.

Remaining Risks: None for this run specifically. [[ISS-005]] still applies if a future step adds LM fusion to decoding.

Related Records: [[EXP-007]], [[VAL-005]], [[VAL-006]], [[ISS-006]], [[ISS-001]]
