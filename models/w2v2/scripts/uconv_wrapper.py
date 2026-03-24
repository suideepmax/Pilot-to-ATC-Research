#!/usr/bin/env python3
import sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--from-encoding', default='utf-8')
parser.add_argument('-t', '--to-encoding', default='utf-8')
parser.add_argument('-x', '--transliterate', default=None)
parser.add_argument('files', nargs='*')
args = parser.parse_args()

def process(text):
    if args.transliterate and 'Lower' in args.transliterate:
        return text.lower()
    return text

if args.files:
    for f in args.files:
        with open(f, 'rb') as fh:
            text = fh.read().decode(args.from_encoding, errors='replace')
        sys.stdout.write(process(text))
else:
    raw = sys.stdin.buffer.read()
    text = raw.decode(args.from_encoding, errors='replace')
    sys.stdout.write(process(text))
