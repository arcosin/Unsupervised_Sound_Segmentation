# Usage: python preprocess.py --in_dir <raw_dir> --out_dir <out_dir> --chunk_time <chunk_time>

import pydub
import os
import argparse
from utils import print_green, print_yellow


def preprocess(in_dir: str = '../dataset/raw/',
               out_dir: str = '../dataset/processed/',
               chunk_time: int = 5) -> None:
    raw_dir = in_dir
    processed_dir = out_dir
    # List all the files ending in .mp3 in the raw directory
    files = sorted([f for f in os.listdir(
        raw_dir) if f.lower().endswith('.wav')])
    print(f'Found {len(files)} files in {raw_dir}')
    print(f'Filenames: {files}')

    # Read from mp3
    # Split the files into 5 second chunks
    # Calculate the number of chunks: total_time / chunk_time

    num_start = 0

    for file in files:
        audio = pydub.AudioSegment.from_wav(os.path.join(raw_dir, file))
        total_time = len(audio) / 1000
        num_chunks = total_time / chunk_time
        num_chunks = int(num_chunks)
        audio_start = 0
        for i in range(num_start, num_chunks + num_start):
            start = audio_start
            end = start + chunk_time * 1000
            chunk = audio[start:end]
            # Create the output file
            if not os.path.exists(processed_dir + file[:-4]):
                os.makedirs(processed_dir + file[:-4])
            chunk.export(processed_dir + file[:-4] +
                         '/' + str(i) + '.wav', format='wav')
            audio_start += chunk_time * 1000
        num_start += num_chunks
        print_green(f'Processed {file} into {int(num_chunks)} chunks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--out_dir', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--chunk_time', type=int,
                        required=True, help='Chunk time in integer seconds')
    args = parser.parse_args()
    preprocess(args.in_dir, args.out_dir, args.chunk_time)
