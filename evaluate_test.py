"""
Evaluate model on test set with ground truth.

    python evaluate_test.py --test_csv ~/asr_project/data/uwb_atcc/manifests/test.csv
"""

import os, argparse, csv, torch, librosa
import numpy as np
import soundfile as sf
import evaluate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=os.path.expanduser(
        "~/asr_project/models/wav2vec2_uwb_atcc_v2"))
    p.add_argument("--test_csv", default=os.path.expanduser(
        "~/asr_project/data/uwb_atcc/manifests/test.csv"))
    p.add_argument("--force_cpu", action="store_true")
    a = p.parse_args()

    device = (torch.device("cpu") if a.force_cpu or not torch.cuda.is_available()
              else torch.device("cuda"))

    proc = Wav2Vec2Processor.from_pretrained(a.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(a.model_dir).to(device)
    model.eval()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    preds, refs = [], []
    with open(a.test_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Evaluating {len(rows)} samples...")
    for i, row in enumerate(rows):
        w, sr = sf.read(row["path"], always_2d=False)
        if w.ndim > 1: w = w.mean(1)
        w = w.astype(np.float32)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)

        inputs = proc(w, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = proc.batch_decode(pred_ids)[0]

        preds.append(text.lower().strip())
        refs.append(row["transcript"].lower().strip())

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(rows)}...")

    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    print(f"\n{'='*40}")
    print(f"Test WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"Test CER: {cer:.4f} ({cer*100:.2f}%)")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()