

# Usage: python wav_to_spectrogram.py --in_file <input_file> --out_file <output_file>

import argparse
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def wav_to_spectrogram(in_file, out_dir):
    # Convert wav to spectrogram without x-axis, y-axis, and colorbar
    # Read the wav file
    data, sr = librosa.load(in_file)
    # Convert to mono
    data = librosa.to_mono(data)
    # Compute the spectrogram
    S = librosa.feature.melspectrogram(
        data, sr=sr, hop_length=1024, n_fft=2048, n_mels=128)
    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)
    # Plot the spectrogram
    fig = plt.figure(figsize=(128, 128))
    librosa.display.specshow(log_S, sr=sr)
    plt.tight_layout()
    plt.axis('off')
    # Save the spectrogram
    out_file = os.path.join(out_dir, os.path.basename(in_file) + '.png')
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,
                        required=True, help='Input wav file')
    parser.add_argument('--out_dir', type=str,
                        required=True, help='Output directory')
    args = parser.parse_args()
    wav_to_spectrogram(args.in_file, args.out_dir)
