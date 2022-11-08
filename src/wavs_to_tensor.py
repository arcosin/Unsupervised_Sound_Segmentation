# Usage: python wavs_to_tensor.py --in_dir <in_dir> --out_dir <out_dir>

import argparse
import os
import librosa
import librosa.display
import tqdm
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils import print_yellow, print_red, print_green


def wav_to_tensor(in_file: str) -> Tensor:
    # Convert wav to spectrogram without x-axis, y-axis, and colorbar
    # Read the wav file
    data, sr = librosa.load(in_file)
    # Convert to mono
    data = librosa.to_mono(data)
    # Compute the spectrogram
    S = librosa.feature.melspectrogram(
        y=data, sr=sr, hop_length=1024, n_fft=2048, n_mels=128)
    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)
    # Convert to tensor
    tensor = torch.from_numpy(log_S)
    return tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--out_dir', type=str,
                        required=True, help='Output directory')
    args = parser.parse_args()

    os.system('cls' if os.name == 'nt' else 'clear')

    # Get all subdirectories in the input directory
    subdirs = [x[0] for x in os.walk(args.in_dir)]

    tensors = []
    # Iterate through all subdirectories, use tqdm to show progress
    with tqdm.tqdm(total=len(subdirs)) as pbar:
        pbar.set_description('Processing')

        for subdir in subdirs:
            # Get all files in the subdirectory
            files = [os.path.join(subdir, f)
                     for f in os.listdir(subdir) if f.endswith('.wav')]
            # Convert all files to tensors
            with tqdm.tqdm(total=len(files)) as pbar2:
                pbar2.set_description(f'Processing {subdir}')
                for file in files:
                    try:
                        tensor = wav_to_tensor(file)
                        tensors.append(tensor)
                    except Exception as e:
                        pass
                    pbar2.update(1)
            os.system('cls' if os.name == 'nt' else 'clear')
            pbar.update(1)

    # Convert all tensors to a single tensor
    tensors = torch.stack(tensors)
    # Save the tensor
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    out_file = args.out_dir + args.in_dir.split('/')[-2] + '.pt'
    print_green(f'Saving {out_file}')
    torch.save(tensors, out_file)
