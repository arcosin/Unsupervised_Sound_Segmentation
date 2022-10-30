python src/preprocess.py \
    --in_dir 'dataset/raw/' \
    --out_dir 'dataset/processed/' \
    --chunk_time 5

# list all folders in the processed folder except spectrogram
for folder in $(ls -d dataset/processed/* | grep -v spectrogram); do
    for wav in $(ls $folder/*.wav); do
        # echo $folder/spectrogram
        python src/wav_to_spectrogram.py \
            --in_file $wav \
            --out_dir $folder/spectrogram/
    done
done