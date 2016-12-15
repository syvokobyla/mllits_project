import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle
import numpy as np
import librosa 

def build_model():
    model = Sequential([
        Dense(1025, input_dim=1025),
        Activation('relu'),
        Dense(1025),
        Activation('linear'),
    ])
    model.compile(loss="MSE", optimizer="rmsprop")
    return model

def fit(model):
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    model.fit(X, Y, batch_size=32, nb_epoch=3)
    return model

def predict(model):

    mix = np.load("mix.sfft.npy").transpose()
    X = np.abs(mix)
    num_examples, num_features = X.shape
    Y = model.predict(X, verbose=True)
    Y = np.minimum(Y, X)

    # Restore phase
    Y_complex = np.exp(1j * np.angle(mix))
    Y_complex *= Y

    result = librosa.core.istft(Y_complex.transpose())
    rate = 16384
    librosa.output.write_wav("predicted.wav", result, sr=rate)

model = build_model()
model = fit(model)
    #model.load_weights("model_weights.keras")
model.save_weights("model_weights.keras")
predict(model)

