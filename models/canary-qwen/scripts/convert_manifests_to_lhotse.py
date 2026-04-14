#!/usr/bin/env python3
"""Convert NeMo JSONL manifests to Lhotse CutSets for SALM training.
Usage: python convert_manifests_to_lhotse.py --data-dir ~/canary-ft/data
"""
import argparse, json
import lhotse
from lhotse import CutSet, MonoCut, SupervisionSegment

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    args = p.parse_args()
    for split in ['train', 'test']:
        print(f'Converting {split}...')
        entries = [json.loads(l) for l in open(f'{args.data_dir}/{split}_manifest.json')]
        cuts = []
        for e in entries:
            fp = e['audio_filepath']
            uid = fp.split('/')[-1].replace('.wav', '')
            cuts.append(MonoCut(id=uid, start=0, duration=e['duration'], channel=0,
                recording=lhotse.Recording.from_file(fp, recording_id=uid),
                supervisions=[SupervisionSegment(id=uid, recording_id=uid,
                    start=0, duration=e['duration'], text=e['text'])]))
        CutSet.from_cuts(cuts).to_file(f'{args.data_dir}/{split}_cuts.jsonl.gz')
        print(f'  {len(cuts)} cuts -> {args.data_dir}/{split}_cuts.jsonl.gz')
    print('Done.')

if __name__ == '__main__': main()
