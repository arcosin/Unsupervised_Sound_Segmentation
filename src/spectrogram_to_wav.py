
# Usage: python spectrogram_to_wav.py --from_array <array_file> --out_file <output_file>

import argparse
import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile as wavfile


def spectrogram_to_wav(from_array: np.ndarray, out_file: str) -> None:
    # Read the spectrogram
    S = np.load(from_array)
    # Convert to linear scale
    S = librosa.db_to_power(S)
    # Convert to wav
    wav = librosa.feature.inverse.mel_to_audio(S)
    # Save the wav
    wavfile.write(out_file, 22050, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_array', type=str,
                        required=True, help='Input spectrogram file')
    parser.add_argument('--out_file', type=str,
                        required=True, help='Output wav file')
    args = parser.parse_args()
    spectrogram_to_wav(args.from_array, args.out_file)
