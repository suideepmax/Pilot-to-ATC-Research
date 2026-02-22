"""
Evaluate model on test set: Greedy vs Beam Search + KenLM.
Produces a full comparison report.

    python evaluate_with_lm.py --test_csv ~/asr_project/data/uwb_atcc/manifests/test.csv \
                               --lm_path ~/asr_project/atc_3gram.arpa
"""

import os, argparse, csv, json, time, torch, librosa
import numpy as np
import soundfile as sf
import evaluate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder

def build_lm_decoder(model_dir, lm_path=None, alpha=0.5, beta=1.5):
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [char for char, idx in sorted_vocab]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        alpha=alpha,
        beta=beta,
    )
    return decoder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=os.path.expanduser(
        "~/asr_project/models/wav2vec2_uwb_atcc_v2"))
    p.add_argument("--test_csv", default=os.path.expanduser(
        "~/asr_project/data/uwb_atcc/manifests/test.csv"))
    p.add_argument("--lm_path", default=None,
                    help="Path to KenLM .arpa or .bin file")
    p.add_argument("--beam_width", type=int, default=100)
    p.add_argument("--lm_alpha", type=float, default=0.5,
                    help="LM weight (higher = more LM influence)")
    p.add_argument("--lm_beta", type=float, default=1.5,
                    help="Word insertion bonus (higher = more words)")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Limit number of test samples (for quick test)")
    p.add_argument("--report_file", default="eval_report.txt")
    p.add_argument("--force_cpu", action="store_true")
    a = p.parse_args()

    device = (torch.device("cpu") if a.force_cpu or not torch.cuda.is_available()
              else torch.device("cuda"))
    print(f"Device: {device}")

    # Load model
    proc = Wav2Vec2Processor.from_pretrained(a.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(a.model_dir).to(device)
    model.eval()
    print(f"Model: {a.model_dir}")

    # Build decoder
    decoder = None
    if a.lm_path:
        decoder = build_lm_decoder(a.model_dir, a.lm_path, a.lm_alpha, a.lm_beta)
        print(f"LM: {a.lm_path}")
        print(f"Beam width: {a.beam_width}, alpha: {a.lm_alpha}, beta: {a.lm_beta}")
    else:
        print("LM: None (greedy only)")

    # Load test data
    with open(a.test_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if a.max_samples:
        rows = rows[:a.max_samples]

    print(f"\nEvaluating {len(rows)} samples...\n")

    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    greedy_preds = []
    beam_preds = []
    refs = []
    errors_greedy = []
    errors_beam = []

    t_start = time.time()

    for i, row in enumerate(rows):
        # Load audio
        w, sr = sf.read(row["path"], always_2d=False)
        if w.ndim > 1:
            w = w.mean(1)
        w = w.astype(np.float32)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)

        inputs = proc(w, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        # Greedy decode
        pred_ids = torch.argmax(logits, dim=-1)
        greedy_text = proc.batch_decode(pred_ids)[0].lower().strip()

        ref = row["transcript"].lower().strip()

        greedy_preds.append(greedy_text)
        refs.append(ref)

        # Per-sample greedy WER
        try:
            sample_wer_g = wer_metric.compute(
                predictions=[greedy_text], references=[ref])
        except Exception:
            sample_wer_g = 1.0

        errors_greedy.append({
            "file": os.path.basename(row["path"]),
            "ref": ref,
            "pred": greedy_text,
            "wer": sample_wer_g,
        })

        # Beam search + LM decode
        if decoder:
            logits_np = logits.cpu().numpy()[0]
            beam_text = decoder.decode(
                logits_np,
                beam_width=a.beam_width,
            ).lower().strip()

            beam_preds.append(beam_text)

            try:
                sample_wer_b = wer_metric.compute(
                    predictions=[beam_text], references=[ref])
            except Exception:
                sample_wer_b = 1.0

            errors_beam.append({
                "file": os.path.basename(row["path"]),
                "ref": ref,
                "pred": beam_text,
                "wer": sample_wer_b,
            })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate
            print(f"  {i+1}/{len(rows)}  ({rate:.1f} samples/s, ETA {eta/60:.1f} min)")

    elapsed = time.time() - t_start

    # -- Compute overall metrics --
    greedy_wer = wer_metric.compute(predictions=greedy_preds, references=refs)
    greedy_cer = cer_metric.compute(predictions=greedy_preds, references=refs)

    beam_wer, beam_cer = None, None
    if decoder and beam_preds:
        beam_wer = wer_metric.compute(predictions=beam_preds, references=refs)
        beam_cer = cer_metric.compute(predictions=beam_preds, references=refs)

    # -- Sort errors by WER (worst first) --
    errors_greedy.sort(key=lambda x: x["wer"], reverse=True)
    if errors_beam:
        errors_beam.sort(key=lambda x: x["wer"], reverse=True)

    # -- Count improvements/regressions --
    improved, regressed, unchanged = 0, 0, 0
    if decoder and beam_preds:
        greedy_by_file = {e["file"]: e for e in errors_greedy}
        beam_by_file = {e["file"]: e for e in errors_beam}
        for fname in greedy_by_file:
            if fname in beam_by_file:
                eg = greedy_by_file[fname]
                eb = beam_by_file[fname]
                if eb["wer"] < eg["wer"]:
                    improved += 1
                elif eb["wer"] > eg["wer"]:
                    regressed += 1
                else:
                    unchanged += 1

    # -- Build report --
    report = []
    report.append("=" * 70)
    report.append("ASR EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"Model:       {a.model_dir}")
    report.append(f"Test set:    {a.test_csv}")
    report.append(f"Samples:     {len(rows)}")
    report.append(f"Time:        {elapsed:.1f}s ({len(rows)/elapsed:.1f} samples/s)")
    report.append("")

    report.append("-" * 70)
    report.append("RESULTS")
    report.append("-" * 70)
    report.append(f"{'Method':<25s} {'WER':>10s} {'CER':>10s}")
    report.append(f"{chr(9472)*25} {chr(9472)*10} {chr(9472)*10}")
    report.append(f"{'Greedy (no LM)':<25s} {greedy_wer*100:>9.2f}% {greedy_cer*100:>9.2f}%")

    if beam_wer is not None:
        report.append(f"{'Beam + KenLM':<25s} {beam_wer*100:>9.2f}% {beam_cer*100:>9.2f}%")
        wer_drop = (greedy_wer - beam_wer) / greedy_wer * 100
        cer_drop = (greedy_cer - beam_cer) / greedy_cer * 100
        report.append("")
        report.append(f"WER reduction:  {wer_drop:+.1f}% relative")
        report.append(f"CER reduction:  {cer_drop:+.1f}% relative")
        report.append("")
        report.append(f"Samples improved:   {improved:>5d} / {len(rows)}")
        report.append(f"Samples regressed:  {regressed:>5d} / {len(rows)}")
        report.append(f"Samples unchanged:  {unchanged:>5d} / {len(rows)}")

    report.append("")
    report.append("-" * 70)
    report.append("WORST 20 SAMPLES (Greedy)")
    report.append("-" * 70)
    for e in errors_greedy[:20]:
        report.append(f"  [{e['file']}] WER={e['wer']*100:.0f}%")
        report.append(f"    REF:  {e['ref']}")
        report.append(f"    PRED: {e['pred']}")
        report.append("")

    if decoder and errors_beam:
        report.append("-" * 70)
        report.append("WORST 20 SAMPLES (Beam + LM)")
        report.append("-" * 70)
        for e in errors_beam[:20]:
            report.append(f"  [{e['file']}] WER={e['wer']*100:.0f}%")
            report.append(f"    REF:  {e['ref']}")
            report.append(f"    PRED: {e['pred']}")
            report.append("")

        report.append("-" * 70)
        report.append("TOP 20 BIGGEST IMPROVEMENTS (Greedy -> Beam + LM)")
        report.append("-" * 70)
        improvements = []
        greedy_by_file = {e["file"]: e for e in errors_greedy}
        beam_by_file = {e["file"]: e for e in errors_beam}
        for fname in greedy_by_file:
            if fname in beam_by_file:
                eg = greedy_by_file[fname]
                eb = beam_by_file[fname]
                delta = eg["wer"] - eb["wer"]
                if delta > 0:
                    improvements.append({
                        "file": fname,
                        "ref": eg["ref"],
                        "greedy": eg["pred"],
                        "beam": eb["pred"],
                        "wer_greedy": eg["wer"],
                        "wer_beam": eb["wer"],
                        "delta": delta,
                    })
        improvements.sort(key=lambda x: x["delta"], reverse=True)
        for imp in improvements[:20]:
            report.append(f"  [{imp['file']}] WER: {imp['wer_greedy']*100:.0f}% -> "
                          f"{imp['wer_beam']*100:.0f}% (delta {imp['delta']*100:+.0f}%)")
            report.append(f"    REF:    {imp['ref']}")
            report.append(f"    GREEDY: {imp['greedy']}"
            report.append(f"    BEAM:   {imp['beam']}")
            report.append("")

    if decoder:
        report.append("-" * 70)
        report.append(f"LM config: alpha={a.lm_alpha}, beta={a.lm_beta}, "
                       f"beam_width={a.beam_width}")
        report.append("-" * 70)

    report.append("=" * 70)

    # Print and save
    full_report = "\n".join(report)
    print(full_report)

    with open(a.report_file, "w") as f:
        f.write(full_report)
    print(f"\nReport saved to: {a.report_file}")

if __name__ == "__main__":
    main()