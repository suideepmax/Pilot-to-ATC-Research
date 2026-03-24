#!/bin/bash
# =============================================================================
# data_prepare_uwb_atcc.sh
# Full data preparation for UWB-ATCC corpus (no sudo required)
# Run from: ~/w2v2-air-traffic
# =============================================================================

set -euo pipefail

ZIP_PATH="${1:-$HOME/Downloads/Air Traffic Control Communication.zip}"
REPO_DIR="${2:-$HOME/w2v2-air-traffic}"

echo "========================================"
echo " UWB-ATCC Data Preparation"
echo " Zip path : $ZIP_PATH"
echo " Repo dir : $REPO_DIR"
echo "========================================"

cd "$REPO_DIR"

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PATH=$HOME/bin:$PATH

# Step 1 - Extract zip
echo "[1/6] Extracting zip..."
unzip -o "$ZIP_PATH" -d data/databases/uwb_atcc/

# Step 2 - Extract rar
echo "[2/6] Extracting rar..."
/home/kotasthane/miniconda3/bin/bsdtar -xf data/databases/uwb_atcc/ZCU_CZ_ATC.rar -C data/databases/uwb_atcc/

# Step 3 - Organize into audio/transcripts
echo "[3/6] Organizing files..."
mkdir -p data/databases/uwb_atcc/ZCU_CZ_ATC/audio
mkdir -p data/databases/uwb_atcc/ZCU_CZ_ATC/transcripts
mv data/databases/uwb_atcc/*.wav data/databases/uwb_atcc/ZCU_CZ_ATC/audio/ 2>/dev/null || true
mv data/databases/uwb_atcc/*.trs data/databases/uwb_atcc/ZCU_CZ_ATC/transcripts/ 2>/dev/null || true

echo "Audio files: $(ls data/databases/uwb_atcc/ZCU_CZ_ATC/audio/ | wc -l)"
echo "Transcript files: $(ls data/databases/uwb_atcc/ZCU_CZ_ATC/transcripts/ | wc -l)"

# Step 4 - Parse trs files manually (process substitution workaround)
echo "[4/6] Parsing trs files to STM format..."
mkdir -p experiments/data/uwb_atcc/prep

rm -f experiments/data/uwb_atcc/prep/text0_stm
for trs in $(find data/databases/uwb_atcc/ZCU_CZ_ATC/transcripts -name '*.trs'); do
  uconv -f cp1250 -t utf8 "$trs" | \
    sed 's:encoding="CP1250":encoding="UTF8":' | \
    sed 's:audio_filename="e2_\(.*\)\.wav":audio_filename="\1":' | \
    python3 data/utils/trs2stm.py /dev/stdin
done >> experiments/data/uwb_atcc/prep/text0_stm

echo "STM lines: $(wc -l < experiments/data/uwb_atcc/prep/text0_stm)"

# Step 5 - Generate wav.scp and text files
echo "[5/6] Generating wav.scp and text files..."
data=experiments/data/uwb_atcc
keyprefix=uwb-atcc
DATA=data/databases/uwb_atcc/ZCU_CZ_ATC
TEXT_NORMALIZATION=data/utils/normalizer/text_normalization_lc.sh
number_expansion=data/utils/number_expansion_english.sh

for wav in $(find $DATA/audio -name '*.wav'); do
    echo "${keyprefix}_$(basename $wav .wav) sox $wav -twav -r16k - remix - |"
done > $data/wav.scp

paste -d ' ' \
  <(cat $data/prep/text0_stm | awk -v prefix=$keyprefix '{ rec=$1; t_beg=$4; t_end=$5; printf("%s_%s_%06d_%06d %s_%s\n", prefix, rec, t_beg*100, t_end*100, prefix, rec); }') \
  <(cut -d ' ' -f4-5 $data/prep/text0_stm) \
  <(cut -d' ' -f7- $data/prep/text0_stm) | \
  tr -s ' ' > $data/prep/text1_raw_spk

paste -d' ' \
  <(cut -d' ' -f1-4 $data/prep/text1_raw_spk) \
  <(cut -d' ' -f5- $data/prep/text1_raw_spk | \
    sed -e 's:(\([^()]*\) ([^)]*)):\1:g; s:(\([^()]*\)([^)]*)):\1:g;' \
        -e "s:´:':g; s:?::g; s:¨::g;" \
        -e 's:6raha:Praha:g; s:0direct:zero direct:g; s:6t:six t:g;') \
  > $data/prep/text2_rem-act-pron

cp $data/prep/text2_rem-act-pron $data/prep/text2_raw_spk

cat $data/prep/text2_rem-act-pron | \
  python3 data/utils/expand_uc_acronyms.py | \
  data/utils/remove_diacritics.sh | \
  sed 's:<sil>::g;' > $data/prep/text3_acron

paste -d' ' \
  <(cut -d' ' -f1-4 $data/prep/text3_acron) \
  <(cut -d' ' -f5- $data/prep/text3_acron | \
    uconv -f utf8 -t utf8 -x "Any-Lower" | \
    sed 's/\[/ \[/g; s/\]/\] /g' | \
    sed -e 's/<[^<>]*>//g' | \
    sed 's:+:-:g;' | \
    $number_expansion | \
    $TEXT_NORMALIZATION) | \
  tr -s ' ' | awk 'NF>4' > $data/prep/text_tags

cut -d' ' -f1,5- $data/prep/text_tags > $data/prep/text_tags2

python3 data/utils/normalizer/final_normalization.py \
  --mapping data/utils/normalizer/words.txt \
  --input $data/prep/text_tags2 \
  --output $data/prep/text_tags3

paste -d' ' \
  <(cut -d' ' -f1-4 $data/prep/text_tags) \
  <(cut -d' ' -f2- $data/prep/text_tags3 | \
    sed 's/ |/_|/g; s/| /|_/g' | \
    uconv -f utf8 -t utf8 -x "Any-Lower" | \
    tr -s ' ') \
  > $data/prep/text_tags_final

python3 data/databases/uwb_atcc/spk_id_tagger.py $data/prep/text_tags_final

cut -d' ' -f1,5- $data/prep/text2_raw_spk > $data/text
awk '{print $1, $2, $3, $4}' $data/prep/text2_raw_spk | sort > $data/segments
awk '{print $1, $2}' $data/prep/text2_raw_spk | sort > $data/utt2spk
cp $data/prep/utt2speakerid $data/

# Step 6 - Train/test split
echo "[6/6] Generating train/test split..."
cut -d' ' -f1 $data/text > $data/ids

python3 data/utils/gen_train_test.py \
  --seed 1234 \
  --train-percentage 80 \
  --input-csv $data/ids

for ds in train test; do
  files_to_filter="text segments utt2spk utt2speakerid"
  for file_to_filter in $files_to_filter; do
    perl data/utils/filter_scp.pl $data/$ds/ids \
      $data/$file_to_filter > $data/$ds/$file_to_filter
  done
  cut -d' ' -f2 $data/$ds/segments > $data/$ds/ids_wav
  perl data/utils/filter_scp.pl $data/$ds/ids_wav \
    $data/wav.scp > $data/$ds/wav.scp
done

echo "========================================"
echo " Data preparation complete!"
echo " Train: $(wc -l < $data/train/text) utterances"
echo " Test : $(wc -l < $data/test/text) utterances"
echo "========================================"
