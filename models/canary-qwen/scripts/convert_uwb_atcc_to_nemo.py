#!/usr/bin/env python3
"""Fixed converter - handles sox pipe commands in wav.scp"""

import json, os, subprocess, argparse

def read_kaldi_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) == 2: data[parts[0]] = parts[1]
    return data

def read_segments(filepath):
    segments = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segments[parts[0]] = {
                    'recording_id': parts[1],
                    'start': float(parts[2]),
                    'end': float(parts[3]),
                    'duration': round(float(parts[3]) - float(parts[2]), 4)
                }
    return segments

def extract_wav_path(wavscp_value, repo_dir):
    """Parse sox pipe command from wav.scp and return absolute path to source wav."""
    # Format: sox data/databases/uwb_atcc/ZCU_CZ_ATC/audio/FILE.wav -twav -r16k - remix - |
    parts = wavscp_value.strip().rstrip('|').split()
    for p in parts:
        if p.endswith('.wav'):
            # Convert relative path to absolute using repo_dir
            if not os.path.isabs(p):
                return os.path.join(repo_dir, p)
            return p
    return None

def extract_segment(wav_path, output_path, start, duration, target_sr=16000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        subprocess.run(['sox', wav_path, '-r', str(target_sr), '-c', '1', '-b', '16',
                        output_path, 'trim', str(start), str(duration)],
                       check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        try:
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-ss', str(start), '-t', str(duration),
                            '-ar', str(target_sr), '-ac', '1', '-acodec', 'pcm_s16le', output_path],
                           check=True, capture_output=True, text=True)
            return True
        except Exception as e2:
            return False

def convert_split(data_dir, output_dir, split, repo_dir, target_sr=16000):
    split_dir = os.path.join(data_dir, split)
    texts = read_kaldi_file(os.path.join(split_dir, 'text'))
    segments = read_segments(os.path.join(split_dir, 'segments'))
    wav_scp = read_kaldi_file(os.path.join(split_dir, 'wav.scp'))

    print(f"  {len(texts)} utterances, {len(segments)} segments, {len(wav_scp)} recordings")

    # Pre-resolve all wav paths
    wav_paths = {}
    for rec_id, cmd in wav_scp.items():
        resolved = extract_wav_path(cmd, repo_dir)
        if resolved and os.path.exists(resolved):
            wav_paths[rec_id] = resolved
        else:
            print(f"  WARNING: cannot resolve wav for {rec_id}: {resolved}")

    print(f"  Resolved {len(wav_paths)}/{len(wav_scp)} wav paths")

    audio_out = os.path.join(output_dir, 'audio', split)
    manifest_path = os.path.join(output_dir, f'{split}_manifest.json')
    os.makedirs(audio_out, exist_ok=True)

    entries, skipped, errors = [], 0, 0
    total = len(texts)
    for i, (utt_id, text) in enumerate(texts.items()):
        if (i+1) % 2000 == 0:
            print(f"  Processing {i+1}/{total}...")

        if not text.strip():
            skipped += 1; continue
        if utt_id not in segments:
            skipped += 1; continue

        seg = segments[utt_id]
        rec_id = seg['recording_id']
        duration = seg['duration']

        if duration < 0.2 or duration > 40.0:
            skipped += 1; continue
        if rec_id not in wav_paths:
            skipped += 1; continue

        seg_file = os.path.join(audio_out, f"{utt_id}.wav")
        if not os.path.exists(seg_file):
            ok = extract_segment(wav_paths[rec_id], seg_file, seg['start'], duration, target_sr)
            if not ok:
                errors += 1; continue

        entries.append({
            "audio_filepath": os.path.abspath(seg_file),
            "text": text.strip(),
            "duration": duration
        })

    with open(manifest_path, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')

    durations = [e['duration'] for e in entries]
    print(f"  Written: {len(entries)} entries -> {manifest_path}")
    print(f"  Skipped: {skipped}, Errors: {errors}")
    if durations:
        print(f"  Duration: {sum(durations)/3600:.2f} hrs, Avg: {sum(durations)/len(durations):.2f}s, Range: {min(durations):.2f}-{max(durations):.2f}s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--repo-dir', required=True, help='Root of w2v2-air-traffic repo (for resolving relative paths)')
    p.add_argument('--target-sr', type=int, default=16000)
    args = p.parse_args()

    print("=" * 60)
    print("UWB-ATCC -> NeMo Manifest Converter (fixed)")
    print(f"  Data dir:  {args.data_dir}")
    print(f"  Repo dir:  {args.repo_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Target SR: {args.target_sr}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    for split in ['train', 'test']:
        if os.path.exists(os.path.join(args.data_dir, split)):
            print(f"\n{split.upper()}:")
            convert_split(args.data_dir, args.output_dir, split, args.repo_dir, args.target_sr)

    print("\nDONE.")

if __name__ == '__main__':
    main()
