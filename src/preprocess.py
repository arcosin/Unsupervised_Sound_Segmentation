# Usage: python preprocess.py --in_dir <raw_dir> --out_dir <out_dir> --chunk_time <chunk_time>

import pydub
import os
import argparse


def preprocess(in_dir='../dataset/raw/', out_dir='../dataset/processed/', chunk_time=5):
    raw_dir = in_dir
    processed_dir = out_dir

    # List all the files ending in .mp3 in the raw directory
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.mp3')]
    print(f'Found {len(files)} files in {raw_dir}')
    print(f'Filenames: {files}')

    # Read from mp3
    # Split the files into 5 second chunks
    # Calculate the number of chunks: total_time / chunk_time

    for file in files:
        audio = pydub.AudioSegment.from_mp3(os.path.join(raw_dir, file))
        total_time = len(audio) / 1000
        num_chunks = total_time / chunk_time
        for i in range(int(num_chunks)):
            start = i * chunk_time * 1000
            end = (i + 1) * chunk_time * 1000
            chunk = audio[start:end]
            # Create the output file
            if not os.path.exists(processed_dir + file[:-4]):
                os.makedirs(processed_dir + file[:-4])
            chunk.export(processed_dir + file[:-4] +
                         '/' + str(i) + '.wav', format='wav')
        print_green(f'Processed {file} into {int(num_chunks)} chunks')

        # TODO: Delete the following two lines after 0012 is processed
        print_yellow(f'Warning: 0012 is not processed due to large file size')
        break


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
