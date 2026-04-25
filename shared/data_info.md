# Datasets

---

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

---

# Dataset: ATCOSIM

## Overview
- **Name**: ATCOSIM (Air Traffic Control Simulation Speech Corpus)
- **Source**: Graz University of Technology (SPSC Lab)
- **URL**: https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html
- **Download**: http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso (~2.5GB ISO)
- **Duration**: ~10 hours
- **Sample Rate**: 32kHz (original), resampled to 16kHz via sox
- **Language**: English (non-native speakers)
- **Accents**: German, Swiss-German, Swiss-French
- **Speakers**: 10 total — 6 male (gm1, gm2, sm1, sm2, sm3, sm4), 4 female (gf1, zf1, zf2, zf3)
- **Recording type**: Close-talk headset microphone, real-time ATC simulations

## Data Splits (seed=1234, 80/20)
- **Train**: 7,660 utterances
- **Test**: 1,916 utterances

## Gender Subsets
| Split | Speakers |
|---|---|
| train_female | zf1, zf2, gf1 |
| test_female | zf3 |
| train_male | sm1, sm2, sm3, sm4 |
| test_male | gm1, gm2 |

## Data Format
### Kaldi format (used by W2V2 pipeline)
Located at: `~/w2v2-air-traffic/experiments/data/atcosim_corpus/{train,test,train_female,test_female,train_male,test_male}/`
- `wav.scp` — recording_id to sox pipeline (on-the-fly 16kHz conversion)
- `segments` — utterance_id, recording_id, start_time, end_time
- `text` — utterance_id and normalized transcription
- `utt2spk` — utterance_id to speaker_id mapping

## Key Differences from UWB-ATCC
| | UWB-ATCC | ATCOSIM |
|---|---|---|
| Duration | ~20.58h | ~10h |
| Sample rate | 8kHz | 32kHz |
| Speech type | Real ATC communications | Simulated ATC communications |
| Speakers | Czech controllers | 10 non-native (DE/CH accents) |
| Gender info | No | Yes (enables gender experiments) |
