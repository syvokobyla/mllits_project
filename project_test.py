import numpy as np
import pickle
from scipy.fftpack import fft
from scipy.io import wavfile
from sklearn import svm

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model

def load_test_data():
    #3661 1025
    X = pickle.load(open('/home/maori/Documents/mllits/Project/test_X.pickle', 'rb'))
    X = np.array(X[:3660]) #TODO: The last array in list does have inconsistent length.
    return X

def test_model(X_test):
    print("Testing the model...")
    model = load_model('my_model.h5')
    Y = model.predict(X_test).astype(np.complex128)
    #Restore phase component (imaginary part of the original input)
    print("Restoring phase component")
    for x, y in zip(X_test, Y):
        y.imag = x.imag

    print("Predict done!")
    return Y

def test_mask_model(X_test):
    print("Testing the mask model...")
    model = load_model('my_model.h5')
    Y = model.predict(X_test).astype(np.complex128)

    res = []
    for x, y in zip(X_test, Y):
        freqs = []
        for freq, mask in zip(x,y):
            if mask == 1:
                freqs.append(freq)
            else:
                freqs.append(0)
        res.append(freqs)
    print("Predict done!")
    return res

def get_freqs_by_mask(X, X_mask):
    res = []
    i = 0
    k = 0
    for x, y in zip(X, X_mask):
        freqs = []
        for freq, mask in zip(x, y):
            if mask == 1:
                freqs.append(freq)
                k += 1
            else:
                i += 1
                freqs.append(0)
        res.append(freqs)
    print("number of skipped freqs", i, "leaved freqs", k)
    return res

def get_wav_from_samples(samples):
    window_len = 2048
    overlap = 512
    overlap_left = overlap // 2
    channel = [] #np.empty(window_len * len(samples))
    #print(samples[0].dtype)
    k = True
    for freqs in samples:
        waveform = np.fft.irfft(freqs)
        waveform = waveform[overlap_left:window_len-overlap_left]
        if k:
            print(len(waveform))
        channel.append(waveform)
    return np.concatenate(channel).astype(np.int16)

X_test = load_test_data()
# Y = test_model(X_test)
#mask = pickle.load(open('/home/maori/Documents/mllits/Project/test_mask.pickle', 'rb'))
#mask = np.array(mask[:3660]) #TODO: The last array in list does have inconsistent length.
#Y = get_freqs_by_mask(X_test, mask)
#Y = test_mask_model(X_test)
wav_data = get_wav_from_samples(X_test)
# wav_data = get_wav_from_samples(Y)
# print(wav_data)
wavfile.write("result_1.wav", 44100, wav_data) #TODO: remove hardcoded rate
