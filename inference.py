"""
Quick inference with the trained Wav2Vec2 model.

    python inference.py --audio path/to/audio.wav
    python inference.py --audio_dir path/to/folder/
"""

import os, argparse, torch, librosa
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_model(model_dir, device):
    proc = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    return proc, model


def transcribe(path, proc, model, device, target_sr=16000):
    w, sr = sf.read(path, always_2d=False)
    if w.ndim > 1:
        w = w.mean(1)
    w = w.astype(np.float32)
    if sr != target_sr:
        w = librosa.resample(w, orig_sr=sr, target_sr=target_sr)

    inputs = proc(w, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    text = proc.batch_decode(pred_ids)[0]
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=os.path.expanduser("~/asr_project/models/wav2vec2_uwb_atcc_v2"))
    p.add_argument("--audio", default=None, help="Single audio file")
    p.add_argument("--audio_dir", default=None, help="Directory of audio files")
    p.add_argument("--force_cpu", action="store_true")
    a = p.parse_args()

    device = torch.device("cpu") if a.force_cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Device: {device}")

    proc, model = load_model(a.model_dir, device)
    print(f"Model loaded: {a.model_dir}")

    files = []
    if a.audio:
        files = [a.audio]
    elif a.audio_dir:
        exts = {".wav", ".flac", ".mp3", ".ogg"}
        files = sorted([os.path.join(a.audio_dir, f)
                        for f in os.listdir(a.audio_dir)
                        if os.path.splitext(f)[1].lower() in exts])
    else:
        print("Provide --audio or --audio_dir")
        return

    print(f"\nTranscribing {len(files)} file(s)...\n")
    for f in files:
        text = transcribe(f, proc, model, device)
        print(f"  {os.path.basename(f):40s} → {text}")


if __name__ == "__main__":
    main()