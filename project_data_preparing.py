import numpy as np
import pickle
from scipy.fftpack import fft
from scipy.io import wavfile
from sklearn import svm


train_X_files = [
    '/home/maori/Documents/mllits/Project/MedleyDB_sample/Audio/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav',
    ]
train_Y_files = [
    '/home/maori/Documents/mllits/Project/MedleyDB_sample/Audio/LizNelson_Rainfall/LizNelson_Rainfall_STEMS/LizNelson_Rainfall_STEM_01.wav',
    ]

test_X_files = [
    '/home/maori/Documents/mllits/Project/MedleyDB_sample/Audio/LizNelson_Coldwar/LizNelson_Coldwar_MIX.wav'
    ]

test_Y_files = [
    '/home/maori/Documents/mllits/Project/MedleyDB_sample/Audio/LizNelson_Coldwar/LizNelson_Coldwar_STEMS/LizNelson_Coldwar_STEM_02.wav'
    ]

def get_samples(filename, window_len, sample_rate=None, overlap=512):
    """Split audio file into number of samples.

    Parameters
    ----------
    filename : str
        Path to the .wav file.
    window_len : int
        The length of window (audio-samples) on which FFT performed.
    sample_rate : int (optional)
        If set, resample sound to this rate.
    overlap : int
        The length of overlapped part.

    Returns
    -------
    sample : list of arrays.
        Each array is a sample's spectrum.
     """
    fs, data = wavfile.read(filename)
    left, right = data[:, 0], data[:,1]
    output = np.empty(len(left), dtype=np.int16)
    samples = []

    overlap_left = overlap // 2
    overlap_right = overlap - overlap_left
    assert overlap_left + overlap_right == overlap

    for n in range(overlap_left, len(left), window_len-overlap):
        sample = left[n-overlap_left:n-overlap_left+window_len]
        window = np.hanning(window_len)
        if len(sample) != window_len:
            continue
        freqs = np.fft.rfft(sample * window)
        if n < 1000 * window_len and n > 800 * window_len: # TODO: remove this and take all samples
            samples.append(freqs)

    return samples

def mask_computing(x, y):
    # computing mask
    sample_mask = []
    for x_freq, y_freq in zip(x, y):
        if x_freq.real < y_freq.real:
            sample_mask.append(0)
        else:
            sample_mask.append(1)
    return sample_mask


def prepare_train_dataset():
    X = []
    Y = []
    mask = []
    my_get_samples = lambda fname: get_samples(fname, window_len=2048)
    skipped = 0
    for X_file, Y_file in zip(train_X_files, train_Y_files):
        for x, y in zip(my_get_samples(X_file), my_get_samples(Y_file)):
            # Skip parts where there is no vocals.
            # TODO: balance vocal vs non-vocal parts
            if np.allclose(y.real, 0):
                skipped += 1
                continue
            X.append(x)
            Y.append(y)
            # computing mask
            mask.append(mask_computing(x, y))

    print("Skipped", skipped)

    print(len(X))
    pickle.dump(X, open('X.pickle', 'wb'))
    pickle.dump(Y, open('Y.pickle', 'wb'))
    pickle.dump(mask, open('mask.pickle', 'wb'))

def prepare_test_dataset():
    X = []
    Y = []
    mask = []
    my_get_samples = lambda fname: get_samples(fname, window_len=2048)
    for X_file, Y_file in zip(test_X_files, test_Y_files):
        for x, y in zip(my_get_samples(X_file), my_get_samples(Y_file)):
            X.append(x)
            Y.append(y)
            mask.append(mask_computing(x, y))
    pickle.dump(X, open('test_X.pickle', 'wb'))
    pickle.dump(Y, open('test_Y.pickle', 'wb'))
    pickle.dump(mask, open('test_mask.pickle', 'wb'))



#get_samples(train_X_files[0], window_len=44100)

prepare_train_dataset()
prepare_test_dataset()
