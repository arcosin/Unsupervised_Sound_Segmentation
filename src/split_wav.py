import librosa
import librosa.display
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import pickle

# File names
FILEPREFIX = "splits_40m"
FILENAMES = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav", "8.wav", "9.wav", "10.wav"]

# Globals
FRAME_SIZE = 500										# Frame size in ms
STEP_SIZE = 250 										# Frame step size for overlap. set to FRAME_SIZE for no overlap
TRAIN_TEST_SPLIT = 0.9									# Train-Test dataset splits
TRAIN_TENSOR_FNAME = "train_tensor_-1to1.pkl"
TEST_TENSOR_FNAME = "test_tensor_-1to1.pkl"

# Parameters
DISPLAY_SPECTOGRAMS = False								# If set to true, will display spectograms while slicing them
NORMALIZE_RANGE = True 									# If set to true, will normalize range to 0-1
DENORM_FACTOR = 63.99771								# Factor to denormalize range

# Functions
def display_spectogram(spec):
	if DISPLAY_SPECTOGRAMS:
		plt.figure()
		librosa.display.specshow(spec)
		plt.colorbar()
		plt.show()

def undo_range_normalization(spec, factor = DENORM_FACTOR):
	if type(spectrogram) == list:
		nspec = [item * DENORM_FACTOR for item in spectrogram]
		return nspec
	else:
		return = spectrogram * DENORM_FACTOR


def generate_tensor_pickles():
	s_time = time.time()
	spectrograms = []

	for file in FILENAMES:
		sound_bytes, sr = librosa.load(f"{FILEPREFIX}/{file}", sr=44100)
		spect = librosa.power_to_db(librosa.feature.melspectrogram(y=sound_bytes, sr=44100, n_fft=2205, hop_length=441))

		np.set_printoptions(threshold=sys.maxsize)
		for i in range(0, spect.shape[1] - FRAME_SIZE, STEP_SIZE):
			t_spect = spect[:,i:(i + FRAME_SIZE)]
			spectrograms.append(t_spect)
			display_spectogram(t_spect)

	if (NORMALIZE_RANGE):
		rmax = np.amax(spectrograms)
		rmin = np.amin(spectrograms)
		scale = 0
		if abs(rmax) > abs(rmin):
			scale = abs(rmax)
		else:
			scale = abs(rmin)
		print(scale)
		spectrograms = spectrograms / scale
		print(np.amax(spectrograms),np.amin(spectrograms))
	exit()
	train_list = spectrograms[0:int(len(spectrograms) * TRAIN_TEST_SPLIT)]
	test_list = spectrograms[int(len(spectrograms) * TRAIN_TEST_SPLIT):]
	train_tensor = torch.Tensor(np.array(train_list))
	test_tensor = torch.Tensor(np.array(test_list))

	pickle.dump(train_tensor, open(TRAIN_TENSOR_FNAME, "wb"))
	pickle.dump(test_tensor, open(TEST_TENSOR_FNAME, "wb"))

	print("Tensors dumped to pickle files")
	print(f"Time Elapsed: {(time.time() - s_time):.3f} seconds")

def load_from_pickle():
	return pickle.load(open(TRAIN_TENSOR_FNAME, "rb")), pickle.load(open(TEST_TENSOR_FNAME, "rb"))

def main():
	generate_tensor_pickles()
	train_t, test_t = load_from_pickle()
	print(train_t.shape, test_t.shape)

if __name__ == '__main__':
	main()