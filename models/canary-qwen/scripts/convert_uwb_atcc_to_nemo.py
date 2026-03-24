#!/usr/bin/env python3
"""
Convert UWB-ATCC Kaldi-format data to NeMo JSONL manifest.

UWB-ATCC data (from w2v2-air-traffic) uses Kaldi-style files:
  - wav.scp:   utterance_id /path/to/recording.wav
  - segments:  utterance_id recording_id start_time end_time
  - text:      utterance_id transcription text here

NeMo SALM expects JSONL manifests:
  {"audio_filepath": "/path/to/audio.wav", "text": "transcription", "duration": 5.2}

Usage:
    python convert_uwb_atcc_to_nemo.py \
        --data-dir ~/w2v2-air-traffic/experiments/data/uwb_atcc \
        --output-dir ~/canary-ft/data \
        --target-sr 16000
"""

import argparse, json, os, subprocess

def read_kaldi_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) == 2: data[parts[0]] = parts[1]
            elif len(parts) == 1: data[parts[0]] = ""
    return data

def read_segments(filepath):
    segments = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 4:
                segments[parts[0]] = {
                    'recording_id': parts[1],
                    'start': float(parts[2]),
                    'end': float(parts[3]),
                    'duration': round(float(parts[3]) - float(parts[2]), 4)
                }
    return segments

def extract_segment(wav_path, output_path, start, duration, target_sr=16000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        subprocess.run(['sox', wav_path, '-r', str(target_sr), '-c', '1', '-b', '16',
                        output_path, 'trim', str(start), str(duration)],
                       check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ss', str(start), '-t', str(duration),
                            '-ar', str(target_sr), '-ac', '1', '-acodec', 'pcm_s16le', output_path],
                           check=True, capture_output=True)
            return True
        except:
            return False

def convert_dataset(data_dir, output_dir, split, target_sr=16000):
    split_dir = os.path.join(data_dir, split)
    texts = read_kaldi_file(os.path.join(split_dir, 'text'))
    print(f"  Loaded {len(texts)} utterances from text")

    segments_path = os.path.join(split_dir, 'segments')
    wavscp_path = os.path.join(split_dir, 'wav.scp')
    segments = read_segments(segments_path) if os.path.exists(segments_path) else {}
    wav_scp = read_kaldi_file(wavscp_path) if os.path.exists(wavscp_path) else {}
    print(f"  Loaded {len(segments)} segments, {len(wav_scp)} wav entries")

    audio_out_dir = os.path.join(output_dir, 'audio', split)
    manifest_path = os.path.join(output_dir, f'{split}_manifest.json')
    os.makedirs(audio_out_dir, exist_ok=True)

    entries, skipped = [], 0
    for utt_id, text in texts.items():
        if not text.strip():
            skipped += 1; continue
        if utt_id in segments:
            seg = segments[utt_id]
            rec_id, duration = seg['recording_id'], seg['duration']
            if duration < 0.2 or duration > 40.0:
                skipped += 1; continue
            if rec_id not in wav_scp:
                skipped += 1; continue

            wav_path = wav_scp[rec_id]
            if '|' in wav_path:
                parts = wav_path.split()
                wav_path = parts[1] if parts[0] == 'sox' else parts[0]

            seg_output = os.path.join(audio_out_dir, f"{utt_id}.wav")
            if not os.path.exists(seg_output):
                if not extract_segment(wav_path, seg_output, seg['start'], duration, target_sr):
                    skipped += 1; continue

            entries.append({
                "audio_filepath": os.path.abspath(seg_output),
                "text": text.strip(),
                "duration": duration
            })
        else:
            skipped += 1; continue

    with open(manifest_path, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')

    durations = [e['duration'] for e in entries]
    print(f"  Written {len(entries)} entries to {manifest_path}")
    print(f"  Skipped {skipped} utterances")
    if durations:
        print(f"  Total: {sum(durations)/3600:.2f} hrs, Avg: {sum(durations)/len(durations):.2f}s, Range: {min(durations):.2f}-{max(durations):.2f}s")

def main():
    p = argparse.ArgumentParser(description='Convert UWB-ATCC to NeMo manifest')
    p.add_argument('--data-dir', required=True,
                   help='Path to UWB-ATCC data (e.g., ~/w2v2-air-traffic/experiments/data/uwb_atcc)')
    p.add_argument('--output-dir', required=True,
                   help='Output directory for manifests and extracted audio')
    p.add_argument('--target-sr', type=int, default=16000,
                   help='Target sample rate (default: 16000, required by Canary)')
    args = p.parse_args()

    print("=" * 60)
    print("UWB-ATCC -> NeMo Manifest Converter")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Target SR:  {args.target_sr} Hz")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'test']:
        if os.path.exists(os.path.join(args.data_dir, split)):
            print(f"\nProcessing {split}:")
            convert_dataset(args.data_dir, args.output_dir, split, args.target_sr)

    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  Train manifest: {args.output_dir}/train_manifest.json")
    print(f"  Test manifest:  {args.output_dir}/test_manifest.json")

if __name__ == '__main__':
    main()
