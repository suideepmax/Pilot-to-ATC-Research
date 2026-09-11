#!/usr/bin/env python3
"""Carve a held-out dev split out of train_cuts.jsonl.gz for S3-B3, grouped by
recording session (not by individual cut), so no session's utterances are
split across train/dev.

Why this exists: salm_uwb_atcc_s3b3_fixed.yaml's validation_ds previously
pointed at test_cuts.jsonl.gz -- the same file used for final WER reporting --
so early-stopping/checkpoint-selection decisions were being made on what
should be a held-out test set (research_log/ISSUES.md, GPT-6 Astra review,
2026-09-10). This script fixes that going forward. It does NOT retroactively
clean already-reported Gate 2/3/ablation results, which already used
test_cuts.jsonl.gz for validation -- that limitation must be disclosed, not
silently erased (see research_log/VALIDATION.md VAL-015/016/017 notes).

Grouping unit: UWB-ATCC cut IDs look like
    uwb-atcc_<POSITION>-<SESSIONHASH>_<start_ms>_<end_ms>_<AT|PI>
e.g. uwb-atcc_APP-lnNSkN_000056_000423_AT. The `<POSITION>-<SESSIONHASH>`
prefix identifies one continuous recorded session; each cut's recording_id
equals its cut id 1:1 (single-cut recordings), so grouping must be done on
this prefix, not on recording_id or speaker (speaker is unset in this
corpus's lhotse cuts; recording_id gives 11543 unique values == unique cuts,
which would not prevent same-session leakage across the split).

Also excludes session uwb-atcc_ACCU-pwnH5N entirely (confirmed via direct
recording-id-prefix comparison against test_cuts.jsonl.gz to be the one
session prefix present in BOTH train_cuts.jsonl.gz and test_cuts.jsonl.gz --
an existing train/test leak, independent of this dev-split work, fixed here
as a byproduct since we're already rewriting the train cuts file).

Usage:
    conda activate canary_ft
    python make_dev_split.py \
        --train-cuts /home/kotasthane/canary-ft/data/train_cuts.jsonl.gz \
        --test-cuts /home/kotasthane/canary-ft/data/test_cuts.jsonl.gz \
        --out-dir /home/kotasthane/canary-ft/data \
        --dev-fraction 0.08 --seed 1234
"""
import argparse
import random
import re
from pathlib import Path

import lhotse

SESSION_PAT = re.compile(r"^(uwb-atcc_[A-Za-z]+-[A-Za-z0-9]+)_\d+_\d+_[A-Z]+$")


def session_of(cut_id: str) -> str:
    m = SESSION_PAT.match(cut_id)
    if not m:
        raise ValueError(f"cut id does not match expected UWB-ATCC pattern: {cut_id!r}")
    return m.group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cuts", required=True)
    p.add_argument("--test-cuts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dev-fraction", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    train_cuts = lhotse.CutSet.from_file(args.train_cuts)
    test_cuts = lhotse.CutSet.from_file(args.test_cuts)

    test_sessions = {session_of(c.id) for c in test_cuts}

    by_session = {}
    for c in train_cuts:
        by_session.setdefault(session_of(c.id), []).append(c)

    session_hours_all = {s: sum(c.duration for c in cuts) / 3600 for s, cuts in by_session.items()}

    leaked = sorted(set(by_session) & test_sessions)
    if leaked:
        print(f"Excluding {len(leaked)} train session(s) that also appear in test_cuts.jsonl.gz: {leaked}")
        for s in leaked:
            del by_session[s]

    sessions = sorted(by_session)  # sort first for determinism, then shuffle with a fixed seed
    rng = random.Random(args.seed)
    rng.shuffle(sessions)

    session_hours = {s: session_hours_all[s] for s in by_session}
    total_hours = sum(session_hours.values())
    target_dev_hours = total_hours * args.dev_fraction

    dev_sessions, dev_hours = [], 0.0
    for s in sessions:
        if dev_hours >= target_dev_hours:
            break
        dev_sessions.append(s)
        dev_hours += session_hours[s]
    dev_sessions = set(dev_sessions)
    train_sessions = set(sessions) - dev_sessions

    dev_cuts = [c for s in dev_sessions for c in by_session[s]]
    new_train_cuts = [c for s in train_sessions for c in by_session[s]]

    assert not (dev_sessions & train_sessions & test_sessions)

    out_dir = Path(args.out_dir)
    dev_path = out_dir / "dev_cuts.jsonl.gz"
    train_path = out_dir / "train_cuts_v2.jsonl.gz"
    lhotse.CutSet.from_cuts(dev_cuts).to_file(dev_path)
    lhotse.CutSet.from_cuts(new_train_cuts).to_file(train_path)

    print(f"Original train: {len(train_cuts)} cuts, {sum(session_hours_all.values()):.3f}h "
          f"({len(by_session) + len(leaked)} sessions incl. {len(leaked)} excluded-leaked)")
    print(f"New train:      {len(new_train_cuts)} cuts, {sum(c.duration for c in new_train_cuts)/3600:.3f}h, "
          f"{len(train_sessions)} sessions -> {train_path}")
    print(f"Dev:            {len(dev_cuts)} cuts, {sum(c.duration for c in dev_cuts)/3600:.3f}h, "
          f"{len(dev_sessions)} sessions -> {dev_path}")
    print(f"Test (untouched): {len(test_cuts)} cuts, {sum(c.duration for c in test_cuts)/3600:.3f}h, "
          f"{len(test_sessions)} sessions -> {args.test_cuts}")


if __name__ == "__main__":
    main()
