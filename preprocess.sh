#!/bin/bash
python src/preprocess.py \
    --in_dir 'dataset/raw/' \
    --out_dir 'dataset/processed/' \
    --chunk_time 5


python src/wavs_to_tensor.py \
    --in_dir 'dataset/processed/' \
    --out_dir 'dataset/'

# list all folders in the processed folder except spectrogram
# for folder in $(ls -d dataset/processed/* | grep -v ZOOM0009); do
#     for wav in $(ls $folder/*.wav); do
#         # echo $folder/spectrogram
#         # run in parallel subprocesses

#         python src/wav_to_spectrogram.py \
#             --in_file $wav \
#             --out_dir $folder/spectrogram/
#         sleep 3
#     done
# done

# copy <dirname>_split<n>/spectrogram/* to dataset/processed/<dirname>/spectrogram/*

# echo $(ls -d dataset/processed/*_split* | grep -v ZOOM0009)
# press any key to continue


# for folder in $(ls -d dataset/processed/*_split* | grep -v ZOOM0009); do

#     target_dir=$(echo $folder | sed 's/_split.*//g')
#     if [ ! -d $target_dir/ ]; then
#         mkdir $target_dir/
#     fi

#     if [ ! -f $target_dir/spectrogram/ ]; then
#         mkdir $target_dir/spectrogram/
#     fi

#     for spectrogram in $(ls $folder/spectrogram/*); do

#         # echo $(echo $folder | sed 's/_split.*//g')/spectrogram/

#         mv $spectrogram $target_dir/spectrogram/
#     done
# done