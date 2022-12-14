

# Usage: python wav_to_spectrogram.py --in_file <input_file> --out_dir <output_dir>

import argparse
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import print_yellow, print_red, print_green


def wav_to_spectrogram(in_file: str, out_dir: str) -> None:
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
    # Plot the spectrogram
    fig = plt.figure(figsize=(646, 167))
    librosa.display.specshow(log_S, sr=sr)
    # plt.tight_layout()
    plt.axis('off')
    # Save the spectrogram
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir + in_file.split('/')[-1][:-4] + '.png'
    print_green(f'Saving {out_file}')
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0, dpi=1)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,
                        required=True, help='Input wav file')
    parser.add_argument('--out_dir', type=str,
                        required=True, help='Output directory')
    args = parser.parse_args()
    wav_to_spectrogram(args.in_file, args.out_dir)
