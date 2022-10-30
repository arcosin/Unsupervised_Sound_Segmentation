

# Usage: python wav_to_spectrogram.py --in_file <input_file> --out_dir <output_directory>

import argparse
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def wav_to_spectrogram(in_file, out_dir):
    # Read the wav file
    fs, data = wavfile.read(in_file)
    # Convert to mono
    data = librosa.to_mono(data)
    # Compute the spectrogram
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128)
    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)
    # Plot the spectrogram
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=fs, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    # Save the spectrogram
    out_file = os.path.join(out_dir, os.path.basename(in_file) + '.png')
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,
                        required=True, help='Input wav file')
    parser.add_argument('--out_dir', type=str,
                        required=True, help='Output directory')
    args = parser.parse_args()
    wav_to_spectrogram(args.in_file, args.out_dir)
