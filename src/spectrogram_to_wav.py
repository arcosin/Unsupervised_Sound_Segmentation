
import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile as wavfile
from torch import Tensor


def spectrogram_to_wav(from_array: np.ndarray, out_file: str) -> None:
    # Read the spectrogram
    if type(from_array) == Tensor:
        from_array = from_array.cpu().numpy()

    # Convert to linear scale
    S = librosa.db_to_power(from_array)
    # Convert to wav
    wav = librosa.feature.inverse.mel_to_audio(S)
    # Save the wav
    wavfile.write(out_file, 44100, wav)

# read pkl files as a tensor


if __name__ == '__main__':
    from utils import read_pkl
    from spectrogram_to_wav import spectrogram_to_wav
    import os

    data = read_pkl('../dataset/test_tensor.pkl')
    i = 250
    filename = f'../dataset/wav/test{i}.wav'
    spectrogram_to_wav(data[i], filename)
