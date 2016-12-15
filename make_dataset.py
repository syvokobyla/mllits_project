import librosa
import numpy as np

def prepare_audio_files():
    target_rate = 16384
    print("Resample mix")
    mix, rate = librosa.load("mix.wav", sr=target_rate)
    print("SFFT")
    mix_freqs = librosa.core.stft(mix)
    np.save("mix.sfft", mix_freqs)

def collect_samples():
    """Save X and Y samples to files. """

    window_size = 2048
    examples_wanted = 1000
    examples_wanted_vocals = examples_wanted // 2
    examples_wanted_non_vocal = examples_wanted - examples_wanted_vocals

    print("Load resampled .wav files")
    x = np.load('mix-16k.pickle.npy')
    y = np.load('vocals-16k.pickle.npy')
    assert len(x) == len(y), "mix and vocals length don't match"

    print("Generate frequency frames")
    x = librosa.core.stft(x)
    y = librosa.core.stft(y)

    print("Pick samples")
    collected_samples = 0
    rng = np.random.RandomState(0)
    X = []
    Y = []
    while collected_samples < examples_wanted:
        pos = rng.randint(len(x))
        sample_x = x[:,pos]
        sample_y = y[:,pos]
        Y.append(sample_y)
        X.append(sample_x)
        collected_samples += 1

    print("Collected {} frequency frames".format(collected_samples))

    # Throw away phase, only keep magnitude
    X = np.stack(np.abs(X))
    Y = np.stack(np.abs(Y))
    np.save("X", X)
    np.save("Y", Y)

    print("All done:")
    print("X.shape = {}".format(X.shape))
    print("Y.shape = {}".format(Y.shape))

if __name__ == '__main__':
    prepare_audio_files()
    #collect_samples()
