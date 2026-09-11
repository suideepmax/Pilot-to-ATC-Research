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

Objective: Verify the EXP-007 female training run actually completed successfully before starting the male run, per [[ISS-006]] (wrapper scripts print "Done training"/"Training complete" even after a crash), using deterministic completion-detection rather than the wrapper's own printed message.

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

---

## VAL-008 — Canary-Qwen N-best generation feasibility (Spike A/B) confirmed via source-code inspection

Date: 2026-09-07
Status: PASS

Objective: Determine whether the installed NeMo speechlm2 `SALM.generate()` supports N-best/beam-search decoding, needed for the decoding-fairness fix (N-best + external KenLM rescoring) proposed in `research_report/IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` Section 8 and refined in `FINAL_RESEARCH_PROGRAM.md` Section 5.

Inputs / Configuration: Installed `canary_ft` conda environment, NeMo 2.8.0rc0, file `nemo/collections/speechlm2/models/salm.py`.

Procedure: Direct `grep`/`Read` of the installed source file (not the GitHub source, the actual on-disk installed package) for `generate`, `num_return_sequences`, `num_beams`, `GenerationConfig`.

Expected Result: Either a custom, closed decoding path with no N-best support (would force the ILME/density-ratio fallback), or a pass-through to a standard, more flexible generation API.

Actual Result: `SALM.generate()` (line 289) accepts an optional `generation_config: GenerationConfig` parameter and, at line 408, calls `self.llm.generate(**generation_inputs, **generation_kwargs, generation_config=generation_config)` — a direct pass-through to the Qwen3-1.7B LLM's standard HuggingFace `generate()` method. The docstring (line 317) explicitly demonstrates `GenerationConfig(do_sample=True, num_beams=5)` as supported usage.

Evidence: `nemo/collections/speechlm2/models/salm.py` lines 289-413 (read in full this session).

Result: PASS — N-best generation (via `GenerationConfig(num_beams=N, num_return_sequences=N)`) is feasible with zero code changes. This resolves the previously-open question from `IEEE_REVIEW_RESPONSE_RESEARCH_PLAN.md` Section 8 in the favorable direction and eliminates the need to consider ILME/density-ratio correction (a much more complex, unimplemented path) for the decoding-fairness fix.

Remaining Risks: Not yet empirically executed (only verified against source) — a single-utterance smoke test confirming actual output shape/content is recommended before scheduling real compute on the decoding-fairness experiment, per the report's Spike A "GO CONDITION."

Related Records: [[EXP-010]], [[DEC-005]]

---

## VAL-009 — Spike D: full Canary-Qwen decoder fine-tuning fits in VRAM on 4×RTX 2080 Ti (11GB) — GO

Date: 2026-09-07
Status: PASS (GO)

Objective: Empirically determine whether unfreezing the full Qwen3-1.7B decoder (removing `^llm\..+$`/`^embed_tokens\..+$` from `freeze_params`, no LoRA block) fits in VRAM under the existing FSDP config (`tensor_parallel_size=1, data_parallel_size=4`), per the Spike D design in `research_report/FINAL_RESEARCH_PROGRAM.md` Section 3/Section 16 (EXP-ID S1-D).

Inputs / Configuration: A new scratch config (`/tmp/.../scratchpad/salm_spike_d_full_decoder_smoke.yaml`, not committed to any repo) derived from `models/canary-qwen/scripts/salm_uwb_atcc_v1.yaml`: `freeze_params` reduced to `perception.preprocessor`/`perception.encoder` only (no `llm`/`embed_tokens`), no `lora:` block (confirmed via `nemo/collections/speechlm2/parts/lora.py:maybe_install_lora` that omitting this key skips PEFT wrapping entirely — the LLM trains directly), `trainer.max_steps: 1`, `limit_train_batches: 1`, `limit_val_batches: 0`. Real UWB-ATCC training data (`~/canary-ft/data/train_cuts.jsonl.gz`) was used, not synthetic data.

Procedure: Launched via `setsid`/`nohup` with explicit `conda activate canary_ft` (per [[ENV-004]]'s detached-launch requirement), `torchrun --nproc_per_node=4`, on the real 4-GPU FSDP topology. First attempt hit an unrelated Lightning config validation error (`val_check_interval` incompatible with `limit_train_batches=1`) — a config bug on the operator's part, not a VRAM signal; fixed by removing the stale `val_check_interval` key and relaunched.

Expected Result: Either a clean single optimizer step (VRAM fits, GO) or `CUDA out of memory` (NO-GO).

Actual Result: `[NeMo I ... optim_setup:231] Parameters | trainable=2033839104 (71.49%) | total=2844823552` — 2.03B trainable (llm 1.7B + embed_tokens 311M, matching the intended full-decoder-unfrozen config exactly), 813M (perception/encoder) correctly remained frozen. Training completed 1/1 step in ~4.95s (`train_step_timing in s=4.950`), `Trainer.fit stopped: max_steps=1 reached`, clean process exit, no `CUDA out of memory`, no traceback, no `ChildFailedError`. `nvidia-smi` post-run confirmed all 4 GPUs returned to idle (0% util, near-zero memory) — no leaked process.

Evidence: `/tmp/.../scratchpad/spike_d_smoke_retry.log` (full training log, this session); direct process/GPU state checks via `pgrep`/`nvidia-smi` before, during, and after.

Result: **PASS — GO.** Full Canary-Qwen decoder fine-tuning (adaptation-scope Branch 3 / EXP-ID S3-B3 in `FINAL_RESEARCH_PROGRAM.md`) is technically feasible on this hardware and should proceed to a real multi-hour run per the report's Stage 3 plan, pending explicit approval to launch that (much longer, ~21 GPU-hour) run.

Limitations: This confirms VRAM fits for a **single step at `limit_train_batches=1`, batch_size=2**. It does not confirm behavior across a full 10,000-step run (e.g., memory fragmentation over time, gradient-accumulation-related peak usage at `accumulate_grad_batches=4`, or checkpoint-saving overhead, none of which were exercised in this 1-step test with checkpointing disabled). A benign warning also appeared: `Parameter freezing patterns UNMATCHED against any parameter: ['^perception\.preprocessor\..+$']` — this regex didn't match any actual submodule name, but did not affect the trainable-parameter split that was actually verified (2.03B trainable, matching llm+embed_tokens exactly).

Related Records: [[EXP-010]], [[VAL-008]], [[DEC-005]], [[ENV-004]]

---

## VAL-010 — Stage 2 fresh Canary-Qwen v1-equivalent control checkpoint: trained, verified, and evaluated — WER matches historical figure

Date: 2026-09-08
Status: COMPLETE, PASS

Objective: Establish a live, freshly-trained, config-verified Canary-Qwen "v1-equivalent" (LoRA q/v-only, no added regularization) checkpoint on UWB-ATCC to serve as the common reference point for the adaptation-scope and regularization studies in `research_report/FINAL_RESEARCH_PROGRAM.md` — Stage 2 / EXP-ID S2-CTRL, since no such checkpoint survived from before (checkpoints for this exact config were confirmed absent, [[ISS-009]]).

Inputs / Configuration: Exact committed `models/canary-qwen/scripts/salm_uwb_atcc_v1.yaml`, deployed unmodified to `~/canary-ft/conf/salm_uwb_atcc.yaml` (byte-identical, confirmed via `diff` before launch). LoRA `target_modules: [q_proj, v_proj]`, r=128, no SpecAugment, `weight_decay=1e-3`, `max_steps=10000`, 4×RTX 2080 Ti FSDP (`tensor_parallel_size=1, data_parallel_size=4`).

Procedure: Launched via `setsid`/`nohup` with explicit `conda activate canary_ft` (per [[ENV-004]]). Monitored continuously via background `Monitor` watches on `val_loss` and completion/failure signatures — no manual polling. Verified completion via real evidence, not the printed message alone: clean process exit (`kill -0`/`pgrep`), 0 failure signatures in the full log, exact log line `` `Trainer.fit` stopped: `max_steps=10000` reached ``, and the final checkpoint (`step=10000-last.ckpt`, 5.5GB, all 4 FSDP shards + metadata present) confirmed on disk. Evaluated via the repo's own `models/canary-qwen/scripts/eval_finetuned.py` (unmodified), single GPU, inference-only, against the full UWB-ATCC test set (2,886 utterances) — the script's built-in NaN-weights check passed (0/1719 tensors NaN) before decoding began.

Expected Result: A WER broadly consistent with the historically-cited 23.32% figure for this exact config, if the pipeline is correctly reproducible.

Actual Result: **WER = 23.3210%** (`{"wer": 0.2332098511353502, "samples": 2886, "errors": 0}`, read directly from the eval script's output JSON, not the printed log) — matches the historically-cited 23.32% to within rounding. Training runtime: ~21 hours wall-clock (10,000 optimizer steps at ~1.69–1.70s/optimizer-step observed throughout, consistent start-to-finish, no slowdowns or stalls). Final training val_loss = 0.80481 (best during training was 0.69849, reached at an earlier checkpoint — the *final* checkpoint is not the *lowest-val_loss* checkpoint, worth noting for any future "best checkpoint" selection decision).

Evidence: `stage2_v1equiv_eval_results.json`; `stage2_canary_v1equiv.log` (full training log, this session); `nvidia-smi`/`pgrep` checks throughout training and at completion; `~/canary-ft/experiments/checkpoints/step=10000-last.ckpt` (verified present, correct size/shard count).

Result: **PASS** — this checkpoint is now the verified live control for Stage 3 (adaptation-scope, regularization) comparisons. Its close match to the historical figure also indirectly increases confidence that the pipeline itself (not just this one run) is sound — a relevant data point for [[ISS-009]]'s open question about the historical v3 checkpoint's provenance, though it does not resolve that question directly since v3 used different regularization settings.

Limitations: Single run, single seed (1234, matching convention) — per the plan's statistical policy ([[EXP-010]]), repeated seeds are reserved for the final headline configuration only, not this control run. Training wall-clock (~21h) is ~4x the ~5.3h documented in the manuscript/repo docs for this exact config — this discrepancy remains unexplained and should be corrected in any redesigned manuscript's Table V; it does not affect the validity of this result, only the resource-planning estimates built on the old figure.

Related Records: [[EXP-010]], [[ISS-009]], [[DEC-005]], [[ENV-004]]

---

## VAL-011 — Historical v3 checkpoint: config verified genuine (via embedded metadata), but WER does not reproduce the cited 20.70%

Date: 2026-09-08
Status: COMPLETE — surprising result, reproducibility gap identified

Objective: Evaluate the surviving historical Canary-Qwen "v3" UWB-ATCC checkpoint ([[ISS-009]]) to obtain a real WER number, and resolve whether its training config actually matched the documented v3 recipe (the question [[ISS-009]] originally raised).

Inputs / Configuration: `~/canary-ft/experiments/checkpoints_v3_HISTORICAL_BACKUP_20260422/step=10000-last.ckpt` (weights protected via directory rename before Stage 2 could overwrite them — timing verified safe). Config provenance re-established via `torch.load(.../meta.pt')['hyper_parameters']['cfg']` — the checkpoint's own embedded hyperparameters, not an external YAML file (which was found to be contaminated — see [[ISS-009]] correction).

Procedure: (1) Loaded `meta.pt` directly with `torch`, extracted `hyper_parameters.cfg`, confirmed it matches the committed `salm_uwb_atcc_v3.yaml` recipe exactly (`lora_dropout=0.1`, `target_modules=[q_proj,v_proj]`, `spec_augment` present, `weight_decay=0.01`). (2) Ran the repo's own `models/canary-qwen/scripts/eval_finetuned.py` (unmodified) against this checkpoint, single GPU, full UWB-ATCC test set (2,886 utterances) — the script's built-in NaN-weights check passed first.

Expected Result: WER close to the manuscript's cited 20.70% (Table I), given the config now confirmed to match.

Actual Result: **WER = 23.3210%** (`{"wer": 0.2332098511353502, "samples": 2886, "errors": 0}`, read from the eval script's output JSON) — matching the plain LoRA-only baseline's WER (also 23.32%, see [[VAL-010]]), essentially identical to it, and **not** the manuscript's cited 20.70% for v3.

Evidence: `historical_v3_eval_results.json`; `meta.pt` embedded hyperparameters (quoted in full in the corrected [[ISS-009]]); `historical_v3_eval.log`.

Result: Config provenance question — **RESOLVED**, config matches documented v3 exactly. WER reproducibility question — **NOT RESOLVED, genuinely surprising**: a checkpoint with the verified-correct v3 training recipe does not reproduce the cited 20.70% figure; it reproduces the *un-regularized* baseline's WER instead, as if regularization had no effect on this specific checkpoint despite being present in its config.

Candidate explanations, none yet verified:
1. This may not actually be the specific checkpoint/run that produced the reported 20.70% figure — a different, still-undiscovered v3 run might exist (the "run_0"/"run_1" directories show at least 2 more historical Canary-Qwen runs existed with parameters not otherwise documented, e.g. `run_1`'s r=64 LoRA rank).
2. The original 20.70% figure may have been computed with a different evaluation methodology than the current `eval_finetuned.py` (different decoding params, different manifest, different checkpoint selection — e.g. best-val-loss checkpoint rather than final step).
3. SpecAugment/regularization, despite being present in the config, may not have actually been effective for some other reason (e.g. a training bug, or the `spec_augment` block being present in config but not actually wired into the forward pass — not yet checked in the perception module's actual `forward()`).
4. The 20.70% figure itself may not be reliably reproducible/correct — this would be a serious finding for the redesigned manuscript.

Limitations: This session only checked *that* the config matches, not whether `spec_augment` was actually applied during the forward pass (candidate explanation 3 is unverified). The `run_0`/`run_1` directories were not checkpoint-recoverable, so their actual results (if any) cannot be independently checked.

**UPDATE (2026-09-08) — HuggingFace check resolves candidate explanation 1, deepens the finding:** the user pointed out (correctly — this was a real gap in the investigation, since README.md/REPLICATION_GUIDE.md were both read earlier this session but never re-consulted for this question) that trained models were uploaded to HuggingFace (`suideepmax/canary-qwen-2.5b-atc-lora`, `suideepmax/canary-qwen-2.5b-atc-unfrozen`, per README.md line 41 and REPLICATION_GUIDE.md lines 256-258). Downloaded and inspected the `canary-qwen-2.5b-atc-lora` repo directly via `huggingface_hub`:
- It contains `v3_results.json` (`wer: 0.2070041846744872` = 20.70%) and `learning_curve_v3.json` — a genuine v3 result record — **but only one `consolidated_model.pt` file (5.85GB)**.
- The repo's own `README.md` **Results table explicitly states the uploaded model is the plain LoRA baseline**: `"Canary-Qwen (LoRA) | 27.8M (0.97%) | 23.32%"`, and its own **Learning Curve table converges to 24.53% at step 10,000** — never approaching 20.70% at any point in training.
- **Conclusion: no v3 model weights were ever uploaded to HuggingFace either.** The `v3_results.json`/`learning_curve_v3.json` files in that repo are orphaned result records from a different run, with no corresponding weights anywhere in the repo.

**SUPERSEDED (2026-09-08) — see [[VAL-012]] for the final, correct resolution.** The "no surviving model" conclusion above was itself an artifact of a second methodology bug (a fixed-path caching collision in `eval_finetuned.py` when two evaluations were run in parallel — see [[VAL-012]]). Once isolated and re-run correctly, the v3 result **is** genuine and reproducible. This record is preserved for the audit trail (it correctly identified real problems — the config-file contamination in [[ISS-009]] — even though its final conclusion was wrong for a different reason). Do not cite the "no verifiable model" conclusion above; see [[VAL-012]].

Related Records: [[ISS-009]], [[VAL-010]], [[VAL-012]], [[EXP-010]], [[DEC-005]]

---

## VAL-012 — FINAL RESOLUTION: v1/v2/v3 all verified genuine; the "23.32%" historical-checkpoint result was a caching-bug artifact

Date: 2026-09-08
Status: COMPLETE, RESOLVED

Objective: Resolve the apparent v3 reproducibility gap raised in [[VAL-011]] by (a) checking whether trained models were uploaded to HuggingFace (a gap in the original investigation — the links were read earlier in this session but not re-checked), and (b) re-running all evaluations in strict isolation after discovering a second methodology bug.

Root cause of the false "23.32%" historical-checkpoint result: `eval_finetuned.py` caches its consolidated model at a **fixed path** (`/tmp/canary_eval_consolidated.pt`) and skips re-consolidation `if os.path.exists(consolidated)`. Two evaluations were run in parallel on separate GPUs (Stage 2's fresh checkpoint, and the historical v3-config checkpoint); the second job found the first job's cache file already present and silently loaded its weights instead of its own. Proof: both runs' WER were bit-for-bit identical to 16 significant figures (0.2332098511353502) — only possible if both loaded the same weights.

Procedure: (1) Deleted the stale `/tmp/canary_eval_consolidated.pt` cache; re-ran the historical checkpoint evaluation in isolation (no concurrent `eval_finetuned.py` job). (2) Independently downloaded `consolidated_model.pt` from both HuggingFace repos referenced in `README.md`/`REPLICATION_GUIDE.md` (`suideepmax/canary-qwen-2.5b-atc-lora`, `suideepmax/canary-qwen-2.5b-atc-unfrozen`) and ran real inference on each with a purpose-written script (`eval_consolidated.py`, no shared-cache risk since it loads directly with no intermediate consolidation step).

Results (all against the full UWB-ATCC test set, 2,886 utterances, verified against output JSON files, not printed logs):

| Source | Result |
|---|---|
| Local historical checkpoint, isolated re-eval | WER = 0.2070041846744872 (20.7004%) |
| HuggingFace `canary-qwen-2.5b-atc-lora` repo, real inference | WER = 0.2070041846744872 (20.7004%) — **bit-for-bit identical** to the local re-eval |
| Author's own `v3_results.json` in that same HF repo | WER = 0.2070041846744872 — matches both of the above exactly |
| HuggingFace `canary-qwen-2.5b-atc-unfrozen` repo, real inference | WER = 0.2382177402757769 (23.8218%) — matches the manuscript's cited 23.82% |
| Stage 2 fresh training + eval, this session ([[VAL-010]]) | WER = 0.2332098511353502 (23.3210%) — the genuine v1/baseline result |

Evidence: `historical_v3_eval_results_CORRECTED.json`, `hf_lora_real_eval_results.json`, `hf_unfrozen_real_eval_results.json`, `stage2_v1equiv_eval_results.json` — all read directly, not from printed logs.

Conclusion: **All three canonical results are genuine and independently verified.** v1 (LoRA baseline) = 23.32%, v2 (encoder unfrozen) = 23.82%, v3 (LoRA + regularization) = 20.70%. The user's original assertion that the 20.70% figure was not fabricated ("I recorded that after finishing training/inference") is fully vindicated with reproducible evidence, not just trust.

**A related, separate, still-unresolved finding surfaced during this investigation**: the `canary-qwen-2.5b-atc-lora` HF repo's own `README.md` **incorrectly** labels the uploaded model as the plain baseline (23.32%, with a learning curve topping out at 24.53%) — but the actual uploaded weights are v3 (20.70%). This README is factually wrong and should be corrected if/when the repo is next touched. Separately, the `canary-qwen-2.5b-atc-unfrozen` repo's own `training_config.yaml` is byte-identical to the v1/baseline config (encoder shown frozen) despite the model measurably, reproducibly differing from v1 (23.82% vs 23.32%) — this is a **third independent instance** of the same "wrong config uploaded alongside correct weights" pattern (see [[ISS-007]], now updated). The true v2 hyperparameters remain unrecoverable from any file; only the WER result is trustworthy.

**Also resolved**: `run_0`/`run_1`/`run_2` (three additional historical training attempts, log-only, no surviving checkpoints) are now fully mapped: `run_0` = lower-LR (1e-4) config, `run_1` = "research-optimized" r=64/4-projection LoRA (crashed with NaN at step 1500, per `PROGRESS.md`, never evaluable), `run_2` = "research-optimized v2" r=128/2-projection LoRA (produced the cited 60.46%). Per user decision (2026-09-08), the research-optimized ablation (both variants) is dropped from the active manuscript/research record — see [[DEC-007]]. The lower-LR config is retained and planned for a fresh, lower-priority retrain — see [[EXP-011]].

Related Records: [[VAL-010]], [[VAL-011]], [[ISS-007]], [[ISS-009]], [[DEC-007]], [[EXP-011]]

## VAL-013 — S4-FAIR: decoding-fairness comparison (N-best + in-domain KenLM) on Canary-Qwen v1

Date: 2026-09-08
Status: COMPLETE, PASS

Objective: Resolve the reviewers' decoding-fairness complaint (manuscript compared W2V2+KenLM against Canary-Qwen native, an apples-to-oranges comparison) by decoding Canary-Qwen with the same in-domain KenLM binary already used for the W2V2 baseline, via 5-best beam-search generation + rescoring, and reporting a fair four-way comparison.

Inputs / Configuration: v1 checkpoint (`step=10000-last.ckpt`, [[VAL-010]]), UWB-ATCC full test set (2,886 utterances), `~/w2v2-air-traffic/experiments/data/uwb_atcc/train/lm/uwb_atcc_4g.binary` (the exact KenLM binary already used for the W2V2+KenLM canonical number, [[DEC-002]]). New scripts: `generate_nbest.py` (5-beam generation via `GenerationConfig(num_beams=5, num_return_sequences=5, output_scores=True, return_dict_in_generate=True)`, confirmed feasible with zero code changes per [[VAL-008]]) and `rescore_kenlm.py` (fuses HF sequence score with KenLM per-word log-score, alpha grid [0.0,0.1,0.3,0.5,0.7,1.0], headline alpha=0.5 chosen to match pyctcdecode's own default used for the W2V2 baseline — not tuned on this test set, to avoid leakage/cherry-picking).

Procedure: Smoke-tested both scripts on 5 samples first (verified real beam diversity, sane per-sequence scores, no exceptions) before running the full test set. Full run: 2,885/2,886 samples succeeded (1 transient CUDA OOM at sample 1777, excluded — 0.03% of test set, not a systematic failure).

Actual Result:
| Config | WER |
|---|---|
| W2V2 native (no LM) | 14.54% |
| W2V2 + KenLM | 12.69% |
| Canary-Qwen v1 native, greedy (existing, [[VAL-010]]) | 23.32% |
| Canary-Qwen v1 native, 5-beam rank-1 | 22.28% |
| Canary-Qwen v1 + KenLM, 5-best rescore, alpha=0.5 (headline) | 21.79% |
| Canary-Qwen v1 + KenLM, best alpha=0.3 | 21.71% |

Evidence: `models/canary-qwen/scripts/nbest_v1_full.json` (raw 5-best + scores), `models/canary-qwen/scripts/kenlm_rescore_v1_results.json` (WER at each alpha).

Conclusion: Giving Canary-Qwen the same in-domain KenLM access as W2V2 closes only ~1.5 points of WER (23.32%→21.79%), and roughly two-thirds of that improvement (23.32%→22.28%) comes from beam search itself, not the external LM. The W2V2-vs-Canary gap (14.54%/12.69% vs ~22%) is barely affected by the fairness fix. This is evidence that the original reviewer-flagged fairness asymmetry, while real, does **not** explain the architecture gap — supporting the manuscript's decoder-adaptation-scope framing rather than undermining it.

Remaining Risks: Alpha was swept but not cross-validated on a held-out split distinct from the test set; the headline alpha=0.5 was chosen by an independent, pre-registered criterion (matching W2V2's own default) specifically to avoid this concern, but a fully rigorous treatment would tune alpha on a dev split.

Related Records: [[VAL-008]], [[VAL-010]], [[DEC-002]]

## VAL-014 — S4-FAIR: decoding-fairness comparison (N-best + in-domain KenLM) on Canary-Qwen v3

Date: 2026-09-08
Status: COMPLETE, PASS (result differs qualitatively from v1 — reported as-is, not adjusted)

Objective: Repeat [[VAL-013]]'s decoding-fairness comparison on the v3 (regularized: LoRA dropout=0.1, SpecAugment, weight_decay=1e-2) checkpoint, since v3 is the manuscript's headline result (20.70%) and the fairness question applies to it independently of v1.

Inputs / Configuration: v3 checkpoint (`~/canary-ft/experiments/checkpoints_v3_HISTORICAL_BACKUP_20260422/step=10000-last.ckpt`, config provenance verified genuine in [[VAL-012]]/[[ISS-009]]), same UWB-ATCC test set and KenLM binary as [[VAL-013]], same `generate_nbest.py`/`rescore_kenlm.py` scripts, unmodified.

Procedure: Smoke-tested on 5 samples first (clean, no exceptions) before the full run. Full run: 2,885/2,886 samples succeeded (1 transient CUDA OOM at sample 1398, excluded — same isolated non-systematic failure mode as [[VAL-013]]).

Actual Result:
| Config | WER |
|---|---|
| Canary-Qwen v3 native, greedy (existing, [[VAL-012]]) | 20.70% |
| Canary-Qwen v3 native, 5-beam rank-1 | 19.42% |
| Canary-Qwen v3 + KenLM, 5-best rescore, alpha=0.5 (headline) | 20.14% |
| Canary-Qwen v3 + KenLM, best alpha in sweep=0.1 | 19.44% |
| Full alpha sweep | 0.0: 19.42% / 0.1: 19.44% / 0.3: 19.76% / 0.5: 20.14% / 0.7: 20.43% / 1.0: 20.75% |

Evidence: `models/canary-qwen/scripts/nbest_v3_full.json`, `models/canary-qwen/scripts/kenlm_rescore_v3_results.json`.

Conclusion: Unlike v1, KenLM rescoring does **not** help v3 — WER increases monotonically as alpha increases past ~0.1, and even the best point in the sweep (alpha=0.1, 19.44%) is a statistical tie with pure beam search (19.42%), not an improvement. All of v3's gain over greedy decoding comes from beam search itself; the external LM adds nothing and the pre-registered headline alpha=0.5 is actually worse than no-LM decoding for this checkpoint. Reported as-is (not replaced with a post-hoc best-alpha number) to avoid cherry-picking. Plausible explanation, not verified: v3's own training already includes SpecAugment/dropout/weight-decay regularization, so its LoRA-adapted decoder may already be better calibrated to in-domain phrasing than v1's, leaving less room for an external n-gram LM to add value — this is a hypothesis, not a tested claim.

Remaining Risks: Same as [[VAL-013]] (alpha not tuned on a separate dev split). The negative result here is more likely to be robust precisely because it does not depend on picking a favorable alpha — it holds across the entire sweep above alpha=0.1.

Related Records: [[VAL-013]], [[VAL-012]], [[VAL-008]]

## VAL-015 — Gate 2 (500-step, production settings) PASS: first confirmed genuine learning on S3-B3's real held-out test set, after ISS-012's clip fix

Date: 2026-09-10
Status: COMPLETE, PASS

Objective: Validate ISS-012's gradient-clipping fix (and, by extension, ISS-011's fp32 master-weight optimizer, never previously tested on a run that took real gradient steps) at a scale beyond tiny diagnostics -- production `salm_uwb_atcc_s3b3_fixed.yaml` settings (`accumulate_grad_batches=8`, full UWB-ATCC train/val data, early stopping enabled with the corrected `min_delta=0.0`), 500 optimizer steps, `warmup_steps=100` (matching the production config's 2000/10000=20% warmup ratio scaled to this budget).

Inputs / Configuration: `models/canary-qwen/scripts/salm_uwb_atcc_s3b3_fixed.yaml` deployed unmodified except CLI overrides `trainer.max_steps=500 model.lr_scheduler.warmup_steps=100`; `salm_train_stable.py`/`master_weight_adamw.py` as committed in `2dc853e` (post ISS-012 fix); 4x RTX 2080 Ti FSDP2; `exp_manager.explicit_log_dir=~/canary-ft/experiments_s3b3_gate2_real/`.

Procedure: Launched via `torchrun --nproc_per_node=4`, monitored to completion (~3h13m wall clock), then independently verified the saved checkpoint (`step=500.ckpt`) by (a) counting nonzero `exp_avg` tensors in the optimizer state and (b) diffing model weights against a freshly-loaded pretrained `Qwen/Qwen3-1.7B` — the same ground-truth method that proved Gate 1 v8/Gate 2 v2/Gate 2 v3 had performed zero updates ([[ISS-012]]).

Expected Result: If the fix works, val_loss should show a real, non-flat improvement trajectory on genuine held-out data (not the frozen-model 12.4-19.9 range every pre-fix gate produced), gradient clipping should engage rarely (well under the 100% zeroing rate ISS-012 found), and the checkpoint diff should show nonzero, non-trivial weight movement.

Actual Result: **All three confirmed.**
- val_loss across 16 validation checkpoints (every 250 batches / ~31 optimizer steps): 4.828 -> 2.975 -> 2.472 -> 2.189 -> 2.038 -> 1.768 -> 1.533 -> 1.342 -> 1.165 -> 1.070 -> 1.005 -> 0.936 -> 0.910 -> 0.887 -> 0.879 -> 0.875 (final, step 500). Monotonic, zero regressions, naturally decelerating -- the classic shape of a real learning curve, not noise. Already inside the same order of magnitude as v1's fully-converged 10000-step val_loss (~0.68) and v3's (~0.58), reached in 5% of the step budget.
- Gradient-clip skip rate: 3/500 steps (0.6%) -- comfortably under Opus's original <1% target, and a categorical improvement over ISS-012's measured 100% zeroing rate on the exact same config pre-fix. Dynamic loss scale calibrated from 1024 down to 128 in the first 4 steps, then held stable with `clip_coef=1` (no clipping needed) for the remainder.
- Checkpoint diff: 311/311 optimizer states have nonzero `exp_avg`; sampled model weights (`llm.model.layers.0.self_attn.q_proj.weight`, `llm.model.layers.15.mlp.down_proj.weight`) show real movement from pretrained init (maxabsdiff 4.6e-4 to 8.5e-4, scaling up consistently from Gate 1 v10's 20-step diff of 1.2e-4, as expected for 25x more steps).

Evidence: `/home/kotasthane/canary-ft/experiments_s3b3_gate2_real/` (full logs, checkpoints, `exp_config.yaml`); direct checkpoint inspection via `torch.distributed.checkpoint.format_utils.dcp_to_torch_save` + `transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B')` diff, same method as [[ISS-012]]'s falsification of the prior three gates.

Result: **PASS**. This is the first run in the entire ISS-011/ISS-012 investigation to demonstrate confirmed, checkpoint-verified, genuine gradient-driven learning at more than a toy scale. Both the fp32 master-weight optimizer (ISS-011/[[DEC-009]]) and the folded-clip fix (ISS-012) are now validated together, not just individually in isolation.

Remaining Risks: Only 500 of the eventual 10000 production steps validated; dynamic loss-scale behavior at later stages (growth back up past 128, and whether skip rate stays low as training progresses and gradients potentially shrink) not yet observed. `BRIDGE_LR=5e-4`'s actual necessity still untested (see planned bridge-LR ablation). Gate 3 (2500 steps) is the next, more decisive validation before committing to the full run.

Related Records: [[ISS-011]], [[ISS-012]], [[ISS-013]], [[DEC-009]]

**Methodology disclosure added 2026-09-10 (GPT-6 Astra external review):** `validation_ds` in this run pointed at `test_cuts.jsonl.gz` -- the same file intended for final WER reporting -- so early-stopping/checkpoint-selection decisions here were made using what should be a held-out test set. Fixed going forward (`make_dev_split.py` carves a proper session-grouped dev split out of train; `salm_uwb_atcc_s3b3_fixed.yaml` now points `validation_ds` at `dev_cuts.jsonl.gz`). This does NOT retroactively clean this specific result -- disclosed, not erased. Also note: this and the checkpoint-cadence gap ([[VAL-017]]'s correction) mean `step=500.ckpt` from this run may not represent a genuinely-monitored best under the corrected checkpoint policy either.

## VAL-016 — Bridge-LR ablation: `BRIDGE_LR=5e-4` genuinely matters, not just a plausible-but-untested optimization

Date: 2026-09-10
Status: COMPLETE, decisive result

Objective: [[ISS-011]] follow-up #3 / [[VAL-015]] left open whether the `perception.proj` bridge-layer LR split (5e-4 vs the decoder body's 1e-5) actually improves outcomes, or was a plausible-but-unproven carryover from v1/v3's LoRA-era hyperparameters. Test directly: identical config/steps/data to [[VAL-015]]'s Gate 2, except `+model.optimizer.bridge_lr=1e-5` (bridge layer trains at the same LR as everything else, disabling the split).

Inputs / Configuration: Identical to [[VAL-015]] (`trainer.max_steps=500`, `model.lr_scheduler.warmup_steps=100`, same real train/val data) except the one added override. Note: the Hydra CLI override initially failed with `Could not override 'model.optimizer.bridge_lr' ... Key 'bridge_lr' is not in struct` -- fixed by using `+model.optimizer.bridge_lr=1e-5` (the `+` prefix is required to add a key not already present in the base config's struct). `configure_optimizers` log line confirmed the fix took effect: `2 bridge params @ lr=1e-05, 309 other trainable params @ lr=1e-05`.

Procedure: Same as [[VAL-015]] -- launch, monitor to completion, independently verify the checkpoint (nonzero `exp_avg` check; skipped the weight-diff-vs-pretrained check here since [[VAL-015]] and Gate 1 v10 already established that check's validity for this exact code path).

Actual Result: Both runs started statistically identically (step-1 val_loss: 4.807 ablation vs 4.828 Gate 2, essentially noise) but diverged sharply as training progressed:

| Checkpoint (opt. steps) | Gate 2 (bridge_lr=5e-4) | Ablation (bridge_lr=1e-5) | Gap |
|---|---|---|---|
| ~31 | 4.828 | 4.807 | -0.02 |
| ~62 | 2.975 | 2.934 | -0.04 |
| ~94 | 2.472 | 2.437 | -0.04 |
| 125 | 2.189 | 2.192 | +0.00 |
| 156 | 2.038 | 2.077 | +0.04 |
| 188 | 1.768 | 2.039 | +0.27 |
| 219 | 1.533 | 1.938 | +0.41 |
| 250 | 1.342 | 1.905 | +0.56 |
| 281 | 1.165 | 1.855 | +0.69 |
| 313 | 1.070 | 1.833 | +0.76 |
| 344 | 1.005 | 1.792 | +0.79 |
| 375 | 0.936 | 1.760 | +0.82 |
| 406 | 0.910 | 1.735 | +0.83 |
| 438 | 0.887 | 1.726 | +0.84 |
| 469 | 0.879 | 1.716 | +0.84 |
| 500 (final) | **0.875** | **1.715** | **+0.84** |

The ablation clearly plateaued in the second half of training (improvements shrank to ~0.01/checkpoint from step ~375 onward, vs Gate 2 which was still improving meaningfully), while Gate 2 kept improving throughout. Final Gate 2 val_loss is roughly half the ablation's. Both runs completed cleanly (no crashes, no divergence) with low, comparable skip rates (Gate 2: 3/500 = 0.6%; ablation: 1/500 = 0.2%) and checkpoint-verified real updates (311/311 nonzero `exp_avg` for both) -- the gap is not an artifact of one run being broken, both trained successfully, one just converges much better.

Evidence: `/home/kotasthane/canary-ft/experiments_s3b3_gate2_real/` and `/home/kotasthane/canary-ft/experiments_s3b3_ablation_bridgelr/` (full logs, checkpoints, exp_configs); direct checkpoint optimizer-state inspection for both.

Result: **BRIDGE_LR=5e-4 is confirmed necessary, not merely plausible.** Consistent with the theoretical argument in ISS-011 follow-up #3 (a randomly-initialized layer needs a proportionally larger LR to escape random init in a comparable number of steps) -- the ablation's plateau is exactly the signature of the bridge layer failing to adapt fast enough at the shared 1e-5 rate, capping how well the rest of the model can compensate. Keep the per-group LR split for Gate 3 and the full production run.

Remaining Risks: Only tested at 500-step scale; whether the gap narrows, holds, or widens over the full 10000-step budget (as the bridge layer eventually reaches a useful state even at 1e-5, ~1800 steps per the original estimate) is not yet known -- Gate 3 (2500 steps) may partially answer this since 2500 > 1800.

Related Records: [[ISS-011]], [[ISS-012]], [[VAL-015]]

**Methodology disclosure added 2026-09-10:** same test-set-as-validation-set issue as [[VAL-015]] applies to both runs in this ablation. The qualitative conclusion (bridge-LR split matters) is a large, consistent effect (2x val_loss gap) unlikely to be an artifact of which split was used for validation, but the exact val_loss numbers above were obtained against `test_cuts.jsonl.gz`, not a proper dev split.

## VAL-017 — Gate 3 (2500-step budget, production settings) PASS: sustained genuine learning, self-terminated cleanly via legitimate early stopping at step 843

Date: 2026-09-10
Status: COMPLETE, PASS

Objective: The decisive staged-validation gate before committing to the full 10000-step production run. Same settings as [[VAL-015]]'s Gate 2 (real train/val data, `accumulate_grad_batches=8`, bridge-LR split at 5e-4 per [[VAL-016]]) but `trainer.max_steps=2500`, `model.lr_scheduler.warmup_steps=500` (same 20% warmup-to-budget ratio as Gate 2 and the eventual full run's 2000/10000).

Inputs / Configuration: `models/canary-qwen/scripts/salm_uwb_atcc_s3b3_fixed.yaml` with CLI overrides `trainer.max_steps=2500 model.lr_scheduler.warmup_steps=500`; `exp_manager.explicit_log_dir=~/canary-ft/experiments_s3b3_gate3/`; early stopping enabled with the corrected `min_delta=0.0`, `patience=3`.

Procedure: Launched, monitored the full run (~6h wall clock until self-termination), independently verified the final checkpoint the same way as [[VAL-015]]/[[VAL-016]] (nonzero `exp_avg` count; weight diff vs freshly-loaded pretrained Qwen3-1.7B).

Actual Result: val_loss trajectory across the run (selected checkpoints, optimizer step / val_loss): 31/8.27 (slower start than Gate 2 by design -- 5x longer warmup), 63/4.50, 94/3.35, 125/2.84, 156/2.53, 188/2.36, 219/2.21, 250/2.15 (25% of budget), 313/1.96, 375/1.71, 438/1.48, 500/1.26 -> 0.97 -> 0.94 (three closely-spaced checkpoints near this boundary), 563/0.92, 625/0.83 (0.791 by the next check), 656/0.781, 687/0.767, 719/0.764, **750/0.754 (TRUE BEST)**. Monotonic throughout up to this point, zero regressions. **Already better than v1's fully-converged 10000-step result is close (v1: ~0.68) despite reaching this point at only 750/10000 = 7.5% of the full production step budget**, though not yet at v1's level.

**CORRECTION (2026-09-10, post-hoc audit prompted by GPT-6 Astra external review):** this record originally stated val_loss=0.754 was "reached at step 843" and described the four pre-stop checkpoints as "0.781, 0.767, 0.764, 0.754... each a small but genuinely nonzero improvement" as if those were the checks immediately before stopping. Both claims were wrong, and the error was caught externally, not by me. Direct audit of the raw training log (`grep "improved\|did not improve"` on `gate3.log`) shows: 0.781/0.767/0.764/0.754 (steps 656/687/719/750) is the tail of the IMPROVING sequence leading up to the true best (0.754 at **step 750**, not 843) -- 843 is where the *stop* was triggered, three validation checks later (~step 781, ~812, ~843). Lightning's EarlyStopping callback only prints a message on an *improving* check ("Metric val_loss improved... New best score: X"); a non-improving check prints nothing at all except the final aggregated stop message. Because of that silent behavior, the actual val_loss values at steps 781/812/843 were never logged anywhere (no persistent per-check metrics existed at the time -- see the `on_validation_epoch_end` JSONL logging fix added 2026-09-10 in `salm_train_stable.py`, which now records every check unconditionally) and are **not recoverable**. The stopping decision itself is still judged legitimate (three genuine non-improving checks after a real best, consistent with `patience=3, min_delta=0.0`) -- what was wrong was this record's narrative of *which* checkpoints were improving and at what step, not whether the callback behaved correctly.

The run **self-terminated via legitimate early stopping** (`Monitored metric val_loss did not improve in the last 3 records. Best score: 0.754.`, printed at ~step 843) -- distinct from the earlier false-positive early-stopping bug (Gate 1 v5, which fired on the very first validation before any real training had occurred) and distinct from the `min_delta` misconfiguration bug (fixed in this session, see [[ISS-012]]'s note and the reverted YAML): this time, `min_delta=0.0` is correct, the model had genuinely plateaued after ~750 real optimizer steps of measurable improvement, and stopping was the intended, correct behavior of the callback -- not a bug. Whether `patience=3` was the *right* policy choice (vs. too aggressive for full-decoder FT's flatter curve) remains open -- see Remaining Risks.

Final checkpoint (`step=843-last.ckpt`) independently verified: 311/311 optimizer states have nonzero `exp_avg`; sampled model weights show real movement from pretrained init (maxabsdiff 1.2e-3 to 1.3e-3, scaling up consistently from Gate 2's 500-step diff of ~4.6e-4 to 8.5e-4, as expected for more steps). Skip rate: 1/843 (0.1%), the lowest of any gate so far, consistent with the dynamic loss scale having more steps to fully calibrate.

Evidence: `/home/kotasthane/canary-ft/experiments_s3b3_gate3/` (full logs, checkpoints including `step=500.ckpt` and `step=843-last.ckpt`, `exp_config.yaml`); direct checkpoint inspection via the same method as [[VAL-015]]/[[VAL-016]]/[[ISS-012]].

Result: **PASS**. This is the strongest evidence yet that the ISS-011/ISS-012 fixes work correctly at the longest scale tested so far (843 real optimizer steps, ~5x longer than Gate 2), with genuine, sustained, checkpoint-verified learning and a clean, correctly-triggered stop rather than a crash or silent failure.

Remaining Risks: The run plateaued at 0.754 rather than continuing toward v1's ~0.68 -- open question whether this reflects (a) a genuine local optimum for this LR/schedule at full-decoder scope, (b) the early-stopping patience (3) being too aggressive for a full-decoder run's flatter improvement curve compared to LoRA's, or (c) something that would resolve with a longer warmup/different LR schedule at the full 10000-step scale. Before launching the full production run, consider whether `patience` should be raised (e.g. to 5-10) given how slowly the curve was still improving near the plateau (0.781->0.767->0.764->0.754 over the last 4 checkpoints -- small but nonzero, arguably not yet a true plateau). This is a scope/methodology decision, not a correctness bug.

**METHODOLOGY DISCLOSURE (2026-09-10, added late -- independent systems-architect audit caught this record was missing what [[VAL-015]]/[[VAL-016]] already have):** `validation_ds` for this run pointed at `test_cuts.jsonl.gz`, the same file intended for final WER reporting -- early-stopping and the recorded best-score value here were computed against what should be a held-out test set, not a proper dev split. This run also trained on the OLD `train_cuts.jsonl.gz`, which included 9 cuts from one session (`uwb-atcc_ACCU-pwnH5N`) later found to also appear in `test_cuts.jsonl.gz` (fixed going forward via `make_dev_split.py`). Disclosed, not erased -- this is the run that produced `step=843-last.ckpt`, the checkpoint being considered for further evaluation/diagnosis, so this disclosure matters more here than for VAL-015/016, not less.

**COMPARABILITY CAVEAT (2026-09-10):** the "already better than v1's ~0.68" framing above is not an apples-to-apples comparison and should not be read as one. v1 trained under plain `torch.optim.AdamW` in fp16 (per [[ISS-011]], numerically degenerate -- `exp_avg_sq` underflows to zero, `weight_decay` inert, update degenerates toward sign-SGD) with LoRA adapters (27.8M trainable params). This run trained under `MasterWeightAdamW` (fp32 master weights, real AdamW math) with the full decoder trainable (~1.4B params). Different optimizer, different parameter count, different effective loss landscape -- the two val_loss numbers are not measuring the same optimization problem, and closeness between them is not evidence of anything about adaptation scope.

Related Records: [[ISS-011]], [[ISS-012]], [[VAL-015]], [[VAL-016]]

## VAL-018 — S3-B3 (`step=843-last.ckpt`) preliminary eval-path smoke: 52.94% WER on 5 samples -- NOT a result, recorded for completeness only

Date: 2026-09-10
Status: INCONCLUSIVE -- recorded to prevent loss, not to be cited as a finding

Objective: The initial run of `eval_finetuned.py --base composed` against Gate 3's checkpoint, done purely to confirm the eval-path fix (ISS-013) loads without error (0 missing/0 unexpected keys, confirmed) -- NOT intended as a WER measurement. Flagged by independent audit as an undocumented number sitting in scratchpad that should either be disclosed with correct caveats or not exist at all; recording it here with those caveats rather than deleting it.

Inputs / Configuration: `--max-samples 5`, `--base composed`, `--exp-config experiments_s3b3_gate3/exp_config.yaml`, checkpoint `step=843-last.ckpt`. Decoding was NOT forced greedy -- at the time this ran, the composed base path backfills Qwen3-1.7B's own `generation_config.json` (`do_sample=True, temperature=0.6, top_k=20, top_p=0.95`), confirmed by independent audit of the actual `transformers` code path. The run's own log was not preserved, so the exact sampling seed/behavior for this specific run cannot be reconstructed.

Actual Result: `{"wer": 0.5294117647058824, "samples": 5, "errors": 0}` (`scratchpad/eval_smoke_step843.json`).

Result: **INCONCLUSIVE, not a measurement.** n=5 with non-deterministic sampled decoding cannot support any claim about model quality in either direction. It is not evidence that S3-B3 works, and not evidence that it doesn't.

Limitations: No log preserved; decoding non-deterministic; sample size far too small for any variance estimate; text normalization not confirmed matched to v1/v3's reporting protocol.

Next Action: superseded by a planned deterministic (forced `do_sample=False`), larger-sample-size re-run (see [[ISS-013]] eval-path items) before any S3-B3 WER is reported or compared against v1/v3.

Related Records: [[ISS-013]], [[VAL-017]]

## VAL-019 — S3-B3 (`step=843-last.ckpt`) first real, deterministic WER: 39.92% (500 samples, greedy) — WORSE than v1/v3 LoRA at this training point

Date: 2026-09-10
Status: COMPLETE, real result (supersedes [[VAL-018]]'s inconclusive 5-sample sampled number)

Objective: Obtain the first trustworthy WER for the full-decoder-FT condition (S3-B3), fixing the decoding non-determinism found by independent audit before any number could be considered comparable to v1/v3's greedy-decoded results.

Fixes applied first: `eval_finetuned.py` now passes `do_sample=False, num_beams=1` as direct keyword arguments to `model.generate()` (not nested inside a `GenerationConfig` object -- a first attempt using the object form was empirically shown NOT to work: two identical runs gave 38.10% then 36.19% WER, proving `transformers`' backfill-from-model-defaults mechanism cannot distinguish an explicit `do_sample=False` from a default `False`, since they're the same value; passing it as a direct `**kwarg` bypasses that ambiguity and was verified deterministic across 2 repeat runs, byte-identical output).

Inputs / Configuration: `--base composed`, `--exp-config experiments_s3b3_gate3/exp_config.yaml`, `--checkpoint step=843-last.ckpt`, `--max-samples 500` (matches this project's own precedent 500-sample eval-subset convention, e.g. [[VAL-003]]), `test_manifest.json` (the real held-out test set, untouched by any split correction).

Procedure: Ran twice at n=10 to confirm determinism (identical WER both times) before committing to the 500-sample run.

Actual Result: `{"wer": 0.39918450560652396, "samples": 500, "errors": 0}` -- **39.92% WER**, 0 generation errors.

Comparison (not a controlled comparison -- see caveats): v1 (LoRA q/v, plain fp16 AdamW, 10000 steps) = 23.32%; v3 (LoRA + regularization, same) = 20.70%. S3-B3 (full-decoder, `MasterWeightAdamW`, early-stopped at 843/10000 steps = 8.4% of budget) = 39.92%. S3-B3 is currently substantially WORSE than both LoRA conditions.

Caveats (do not over-read this number without these):
- S3-B3 stopped at 8.4% of its configured step budget (early stopping on val_loss plateau, [[VAL-017]]) -- this is not a converged result, and no comparably-early-stopped v1/v3 checkpoint exists to compare against.
- Different optimizer (plain fp16 AdamW for v1/v3, numerically degenerate per [[ISS-011]], vs. fp32 `MasterWeightAdamW` for S3-B3) -- confounds any "adaptation scope" interpretation, per [[ISS-013]]/[[ISS-014]] audit notes.
- Random bridge AND mismatched (unadapted) frozen encoder vs. the released model, per [[ISS-013]]'s extended finding -- a materially different speech front-end than what any released-model comparison would use.
- Text normalization not yet confirmed identical to v1/v3's reporting protocol (both lowercase+strip here; v1/v3's exact historical normalization not re-verified in this pass).
- Evaluated on the OLD `test_manifest.json`/`test_cuts.jsonl.gz` (same file used for all of v1/v2/v3's historical numbers) -- this IS the correct comparison set (untouched by the dev-split fix), not a leakage concern for this specific number.

Related diagnostic (same session): [[VAL-020]] (audio-grounding check) confirms this number is not an artifact of the model ignoring its audio input.

Result: **A real, reportable number, but not yet a fair "adaptation scope" comparison point.** It answers "does the current S3-B3 checkpoint, as-is, beat the LoRA baselines" (no), not "does full-decoder adaptation scope help ATC ASR" (still open, confounded by the differences listed above).

Related Records: [[ISS-011]], [[ISS-013]], [[ISS-014]], [[VAL-017]], [[VAL-018]], [[VAL-020]]

## VAL-020 — Teacher-forced audio-grounding check on S3-B3: model IS acoustically grounded, not just language-modeling plausible ATC phrasing

Date: 2026-09-10
Status: COMPLETE, PASS

Objective: Determine whether S3-B3's decreasing training/val loss reflects genuine acoustic grounding or could instead be explained by the model learning ATC-domain language modeling (plausible phrasing) independent of the actual audio content -- raised as a live concern before trusting any WER number from this checkpoint. Per independent audit: a teacher-forced audio-permutation test is cheaper and more decisive than a generation-based audio-mismatch test (no sampling noise, no confound from mismatched-audio changing prompt length).

Inputs / Configuration: New script `models/canary-qwen/scripts/audio_grounding_check.py`. Reuses the project's own `DataModule`/`SALMDataset` pipeline (not a hand-rolled substitute) against `step=843-last.ckpt`'s own `exp_config.yaml` validation_ds (`uwb_atcc_test`, the same set VAL-017 validated against). 6 batches of batch_size=4 (limited by 11GB VRAM for concurrent conformer-encoder attention over multiple audio streams; batch_size=16 OOM'd).

Procedure: For each batch, compute the model's own training-time cross-entropy loss and next-token accuracy with (a) the batch's real audio-to-transcript pairing, and (b) the audio tensor permuted within the batch via a derangement (no fixed points; text/loss_mask untouched). Inference-only, `torch.no_grad()`, no generation, no sampling, fully deterministic.

Actual Result (mean over 6 batches, 54-87 target tokens each):
- Correct-audio: loss=0.6277, token accuracy=0.8689
- Permuted-audio: loss=3.1911, token accuracy=0.4698
- Delta: loss +2.56 (≈5x worse), accuracy -0.40 (≈46% relative drop)

Consistent in direction and magnitude across all 6 batches individually (loss increase ranged +2.34 to +2.79; accuracy drop ranged -0.33 to -0.48) -- not a fluke of one batch.

Result: **PASS.** The model's loss and token-accuracy are strongly and consistently sensitive to which audio it receives. This is not proof of correct word-level acoustic grounding (e.g. it doesn't test whether specific callsigns/numbers are grounded vs. approximately-right phrasing), but it does rule out the concrete concern that the model's low training loss is audio-independent language modeling. VAL-019's 39.92% WER should be read as a real (if currently mediocre) ASR result, not a symptom of a broken or audio-blind pipeline.

Limitations: Ran against `uwb_atcc_test` (the leaked-into-validation set per [[VAL-017]]'s disclosure) since that's what the checkpoint's own frozen `exp_config.yaml` points at -- appropriate here since this diagnostic makes no model-selection decision, only probes an existing checkpoint's behavior. Did not test word-level substitution of specific critical content (callsigns/numbers) -- that remains a distinct, not-yet-done analysis (see decision memo's Candidate C).

Related Records: [[VAL-017]], [[VAL-019]], [[ISS-013]]
