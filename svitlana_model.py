import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle
import numpy as np
import librosa
import mir_eval.separation

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
    print(X.shape)
    model.fit(X, Y, batch_size=32, nb_epoch=3)
    return model

def predict(model, mix_freqs):

    #mix = np.load("mix.stft.npy")#.transpose()
    X = np.abs(mix_freqs)
    num_examples, num_features = X.shape
    Y = model.predict(X, verbose=True)
    Y = np.minimum(Y, X)

    # Restore phase
    Y_complex = np.exp(1j * np.angle(mix_freqs))
    Y_complex *= Y

    return Y_complex

    #result = librosa.core.istft(Y_complex.transpose())
    #rate = 16384
    #librosa.output.write_wav("predicted.wav", result, sr=rate)

    #np.save("predicted", result)

    #source_vocal = librosa.core.istft(np.load("vocal.stft.npy").transpose())
    #mix = librosa.core.istft(np.load("mix.stft.npy").transpose())

    #librosa.output.write_wav("temp_source.wav", source_vocal, sr=rate)
    #print(source_vocal.shape, result.shape)



def evaluate():
    #source_vocal = librosa.core.istft(np.load("vocal.stft.npy").transpose()) #wRONG!!!!!!!!
    predicted_vocal = np.load("predicted.npy")
    predicted_instrument =
    vocal =
    instrument =
    #mix = librosa.core.istft(np.load("mix.stft.npy").transpose())


    #mir_eval.separation.validate(source, res)
    mir_eval.separation.bss_eval_sources([vocal, instrument], [predicted_vocal, predicted_instrument])

    #mir_eval.separation.validate(np.array([source_vocal]), np.array([result]))
        #(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(source_vocal, result)
    #print(sdr, sir, sar, perm)
    #(array([-0.85175804, -0.85175804]), array([ 252.13302926,  252.13302926]), array([-0.85175804, -0.85175804]), array([0, 1]))
    #[-0.85175804] [ inf] [-0.85175804] [0] - mix vs vocal
    #[ 2.61667555] [ inf] [ 2.61667555] [0] - source vocal vs result
    #[ 269.04395712] [ inf] [ 269.04395712] [0] - siurce vocal vs source vocal

    # source vs rsult (array([ 2.61667555,  2.61667555]), array([ 223.19718976,  223.19718976]), array([ 2.61667555,  2.61667555]), array([0, 1]))
    #source vs source([ 269.04395712,  269.04395712]), array([ 266.65785807,  266.65785807]), array([ 263.6174841,  263.6174841]), array([0, 1]))


def _any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))



model = build_model()
model = fit(model)

mix_freqs = np.load("test_mix.stft.npy")
predicted_vocal_freqs = predict(model, mix_freqs)

predicted_vocal_samples = librosa.core.istft(predicted_vocal_freqs.transpose())
rate = 16384
librosa.output.write_wav("predicted.wav", predicted_vocal_samples, sr=rate)

predicted_instrument_samples =

source_vocal =
source_instrument =

sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
                    [vocal, instrument],
                    [predicted_vocal_samples, predicted_instrument_samples]
                    )


#model.load_weights("model_weights.keras")
#model.save_weights("model_weights.keras")
#predict(model)
#evaluate()
