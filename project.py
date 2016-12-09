import numpy as np
import pickle
from scipy.fftpack import fft
from scipy.io import wavfile
from sklearn import svm


train_X_files = [
    'MedleyDB_sample/Audio/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav',
    ]
train_Y_files = [
    'MedleyDB_sample/Audio/LizNelson_Rainfall/LizNelson_Rainfall_STEMS/LizNelson_Rainfall_STEM_01.wav',
    ]

WINDOW_SAMPLES = 44100


def get_samples(filename):
    fs, data = wavfile.read(filename)
    left, right = data[:, 0], data[:,1]
    output = np.empty(len(left), dtype=np.int16)

    samples = []

    for n in range(0, len(left), WINDOW_SAMPLES):
        sample = left[n:n+WINDOW_SAMPLES]
        freqs = np.fft.rfft(sample)

        samples.append(freqs)

    return samples

def prepare_dataset():
		X = []
		Y = []

		for X_file, Y_file in zip(train_X_files, train_Y_files):
			X += get_samples(X_file)
			Y += get_samples(Y_file)

		pickle.dump(X, open('X.pickle', 'wb'))
		pickle.dump(Y, open('Y.pickle', 'wb'))


def train():
	X = pickle.load(open('X.pickle', 'rb'))
	Y = pickle.load(open('Y.pickle', 'rb'))

	clf = svm.SVC()
	clf.fit(
