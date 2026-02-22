"""
Inference with n-gram LM beam search decoding.
Requires: pip install pyctcdecode kenlm

    python inference_with_lm.py --audio ~/asr_project/data/uwb_atcc/audio/test_000000.wav
    python inference_with_lm.py --audio_dir ~/asr_project/data/uwb_atcc/audio/ --lm_path ~/asr_project/atc_3gram.arpa
"""

import os, argparse, json, torch, librosa
import numpy as np
import soundfile as sf
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


def load_model(model_dir, device):
    proc = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    return proc, model


def transcribe(path, proc, model, decoder, device, beam_width=100, target_sr=16000):
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

    # Greedy (no LM)
    pred_ids = torch.argmax(logits, dim=-1)
    greedy_text = proc.batch_decode(pred_ids)[0]

    # Beam search (with LM if provided)
    logits_np = logits.cpu().numpy()[0]
    beam_text = decoder.decode(logits_np, beam_width=beam_width)

    return greedy_text, beam_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=os.path.expanduser(
        "~/asr_project/models/wav2vec2_uwb_atcc_v2"))
    p.add_argument("--lm_path", default=None,
                    help="Path to KenLM .arpa or .bin file (optional)")
    p.add_argument("--lm_alpha", type=float, default=0.5)
    p.add_argument("--lm_beta", type=float, default=1.5)
    p.add_argument("--beam_width", type=int, default=100)
    p.add_argument("--audio", default=None)
    p.add_argument("--audio_dir", default=None)
    p.add_argument("--force_cpu", action="store_true")
    a = p.parse_args()

    device = (torch.device("cpu") if a.force_cpu or not torch.cuda.is_available()
              else torch.device("cuda"))
    print(f"Device: {device}")

    proc, model = load_model(a.model_dir, device)
    decoder = build_lm_decoder(a.model_dir, a.lm_path, a.lm_alpha, a.lm_beta)
    print(f"Model loaded: {a.model_dir}")
    print(f"LM: {a.lm_path or 'None (greedy beam search)'}")

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
    print(f"{'File':40s} | {'Greedy':50s} | {'Beam Search':50s}")
    print("-" * 145)
    for f in files:
        greedy, beam = transcribe(f, proc, model, decoder, device, a.beam_width)
        print(f"  {os.path.basename(f):40s} | {greedy:50s} | {beam:50s}")


if __name__ == "__main__":
    main()