#!/usr/bin/env python
import argparse
import glob
import os

import numpy as np
import mir_eval.separation
import librosa
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers


def build_model(lr=0.001):
    """Build a fully connected NN model (1025x1025x1025).
    
    Parameters
    ----------
    lr : float
        Learning rate.

    Return
    ------
    model : keras.models.Sequential
    """
    model = Sequential([
        Dense(1025, input_dim=1025),
        Activation('relu'),
        Dense(1025),
        Activation('linear'),
        ])
    optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="MSE", optimizer=optimizer)
    return model


def train(model, batch_size=32, epochs=30, x_file="train_mix.stft.npy", \
                                           y_file="train_vocal.stft.npy"):
    """Fit the model.

    Input features must be stored in X.npy file, target in Y.npy.

    Parameters
    ----------
    model : keras.models.Sequential
        Built sequential NN model.
    batch_size : int
        Number of examples in one batch.
    epochs : int
        Number of epochs to train.
    x_file : string
        Filename of x data.
    y_file : string
        Filename of y data.

    Return
    ------
    model : keras.models.Sequential
        Trained NN model.
    """
    X = np.load(x_file)
    Y = np.load(y_file)
    model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs)
    return model


def predict(model, source_mix_freqs, predicted_vocal_fname):
    """Extract vocal freqs from mix freqs.
    
    Parameters
    ----------
    model : keras.models.Sequential
        Trained model.
    source_mix_freqs : string
        File with source mix freqs (.stft file).
    predicted_vocal_fname : string
        File name for resulting wav data.
    
    Return
    ------
    predicted_vocal_samples : list of comlex
        Predicted vocal samples.
    predicted_instrum_samples : list of complex
        Predicted instrument samples.
    """
    mix_freqs = np.load(source_mix_freqs)
    X = np.abs(mix_freqs)
    num_examples, num_features = X.shape
    Y = model.predict(X, verbose=True)
    Y = np.minimum(Y, X)
    # Restore phase
    predicted_vocal_freqs = np.exp(1j * np.angle(mix_freqs))
    predicted_vocal_freqs *= Y
    predicted_instrum_freqs = np.array([x - y for x, y in \
                                        zip(mix_freqs, predicted_vocal_freqs)])
    # Make predicted wav file
    predicted_vocal_samples = librosa.core.istft(predicted_vocal_freqs.transpose())
    predicted_instrum_samples = librosa.core.istft(predicted_instrum_freqs.transpose())
    librosa.output.write_wav(predicted_vocal_fname, predicted_vocal_samples, sr=16384)
    return predicted_vocal_samples, predicted_instrum_samples


def evaluate(predicted_vocal, predicted_instrum, source_audio_path):
    """Evaluate the the separation quality. 
    
    Parameters
    ----------
    predicted_vocal : 
        Predicted vocal samples.
    predicted_instrum :
        Predicted instrument samples
    source_audio_path :
       Path to compared wav files.
    
    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in the mean SIR sense 
    """
    find_files = lambda filename: glob.glob(os.path.join(source_audio_path, '*', filename))
    source_vocal_paths = find_files('vocal.wav')
    source_mix_paths = find_files('mix.wav')
    source_instrument_paths = find_files('instrument.wav')
    
    source_vocal_samples, rate = librosa.load(source_vocal_paths[0], sr=16384)
    source_instrument_samples, rate = librosa.load(source_instrument_paths[0], sr=16384)
    
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
                        np.array([source_vocal_samples, source_instrument_paths]),
                        np.array([predicted_vocal, predicted_instrum])
                        )
    return sdr, sir, sar, perm


def add_train_options(subparsers):
    """Create a subparser for `train` command. """

    parser = subparsers.add_parser('train')
    parser.set_defaults(func=cmd_train)
    parser.add_argument('--batch-size',
        type=int,
        default=32,
        help="Batch size, default is %(default)s")
    parser.add_argument('--epochs',
        type=int,
        default=30,
        help="Number of epochs to train, default is %(default)s")
    parser.add_argument('--lr',
        type=float,
        default=0.001,
        help="Initial learning rate")
    parser.add_argument('--output',
        default="model.keras",
        help="Output model file.")
    parser.add_argument('--source-audio-path', 
        default='../MedleyDB_sample/Test_Audio',
        help="Path to validation audio path, default is %(default)s")


def add_predict_options(subparsers):
    parser = subparsers.add_parser('predict')
    parser.set_defaults(func=cmd_predict)
    parser.add_argument(
        'input',
        help='Path to the extracted audio-mix features (.stft file)')
    parser.add_argument(
        'output',
        help='Path to the output .wav file with extracted vocals')


def cmd_train(args):
    model = build_model(args.lr)
    model = train(model, args.batch_size, args.epochs)
    model.save_weights(args.output)


def cmd_predict(args):
    model = ... # load_model
    predicted_vocal_samples, predicted_instrum_samples = \
        predict(model, args.input, args.output)
    evaluate(predicted_vocal_samples, predicted_instrum_samples,
            args.source_audio_path, rate=16384)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_train_options(subparsers)
    add_predict_options(subparsers)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

