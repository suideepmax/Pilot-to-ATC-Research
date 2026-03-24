# Dataset: UWB-ATCC

## Overview
- **Name**: UWB-ATCC (Air Traffic Control Communications)
- **Source**: University of West Bohemia
- **URL**: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0
- **Duration**: ~20.58 hours
- **Sample Rate**: 8kHz (original), 16kHz (resampled for Canary)
- **Language**: English (Czech ATC controllers)
- **Location**: Prague Airport

## Data Splits (seed=1234, 80/20)
- **Train**: 11,543 utterances, 2,086 recordings
- **Test**: 2,886 utterances, 570 recordings

## Data Formats
### Kaldi format (used by W2V2 pipeline)
Located at: `~/w2v2-air-traffic/experiments/data/uwb_atcc/{train,test}/`
- `wav.scp` — recording_id to wav path mapping
- `segments` — utterance_id, recording_id, start_time, end_time
- `text` — utterance_id and transcription

### NeMo JSONL manifest (used by Canary pipeline)
Located at: `~/canary-ft/data/`
- `train_manifest.json` — one JSON per line: {audio_filepath, text, duration}
- `test_manifest.json` — same format
- `audio/{train,test}/` — individual 16kHz WAV segments
