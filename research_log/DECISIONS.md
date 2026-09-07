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
