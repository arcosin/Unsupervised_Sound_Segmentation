import librosa
import librosa.display
import numpy as np
import soundfile

num_fft=512
def spectrogram_to_wav(from_array: np.ndarray, out_file: str, sr=22050) -> None:
    a = np.exp(from_array) - 1
    p = 2 * np.pi * np.random.random_sample(from_array.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=num_fft))
    soundfile.write(out_file, x, samplerate=sr)


if __name__ == '__main__':
    i = 1
    with open(f'../VQVAE/sounds/x{i}_orig.npy', 'rb') as f:
        spectrogram_orig = np.load(f)
    with open(f'../VQVAE/sounds/x{i}_new.npy', 'rb') as g:
        spectrogram_new = np.load(g)

    spectrogram_orig = spectrogram_orig
    spectrogram_to_wav(spectrogram_orig, f'../VQVAE/sounds/x{i}_orig.wav')

    spectrogram_new = spectrogram_new
    spectrogram_to_wav(spectrogram_new, f'../VQVAE/sounds/x{i}_new.wav')
