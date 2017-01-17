import librosa
import numpy as np

def prepare_audio_files(paths):
    # paths = list((vocal_file, instrument_file, mix_file))
    target_rate = 16384
    print("Resample files")
    files_mix_freqs = []
    files_vocal_freqs = []
    for vocal_file, mix_file in paths:
        mix, rate = librosa.load(mix_file, sr=target_rate)
        vocal, rate = librosa.load(vocal_file, sr=target_rate)

        print("STFT")
        mix_freqs = np.stack(librosa.core.stft(mix), axis=1) # (..., 1025)
        vocal_freqs = np.stack(librosa.core.stft(vocal), axis=1)

        concat = lambda x, y: np.concatenate((x, y), axis = 0 if x else x = y

        files_vocal_freqs = concat(files_vocal_freqs, vocal_freqs)
        files_mix_freqs = concat(files_mix_freqs, mix_freqs)

    return files_mix_freqs,

    np.save("mix.stft", files_mix_freqs)
    np.save("vocal.stft", files_vocal_freqs)

def prepare_freq_file(paths, freq_file_path):
        target_rate = 16384
        print("Resample files")
        files_freqs = []
        for f in paths:
            sample, rate = librosa.load(f, sr=target_rate)
            print("STFT")
            freqs = np.stack(librosa.core.stft(sample), axis=1) # (..., 1025)

            if not files_freqs:
                files_freqs = np.concatenate((files_freqs, freqs), axis = 0)
            else:
                files_freqs = freqs

        np.save(freq_file_path, files_freqs)

def collect_samples(x_file_name, y_file_name):
    """Save X and Y samples to files. """

    #window_size = 2048
    #examples_wanted = 1000
    #examples_wanted_vocals = examples_wanted // 2
    #examples_wanted_non_vocal = examples_wanted - examples_wanted_vocals

    print("Load resampled stft files")
    x = np.load(x_file_name) #'mix.stft.npy')
    y = np.load(y_file_name) #'vocal.stft.npy')
    assert len(x) == len(y), "mix,vocal and instrument lengths don't match"


    print("Pick samples")
    collected_samples = 0
    rng = np.random.RandomState(0)
    X = []
    Y = []
    examples_wanted = len(x)
    while collected_samples < examples_wanted:
        pos = rng.randint(len(x))
        #sample_x = x[:,pos]
        sample_x = x[pos]
        #sample_y = y[:,pos]
        sample_y = y[pos]
        Y.append(sample_y)
        X.append(sample_x)
        collected_samples += 1

    print("Collected {} frequency frames".format(collected_samples))

    # Throw away phase, only keep magnitude
    X = np.stack(np.abs(X))
    Y = np.stack(np.abs(Y))

    print(X.shape)
    np.save("X", X)
    np.save("Y", Y)

    print("All done:")
    print("X.shape = {}".format(X.shape))
    print("Y.shape = {}".format(Y.shape))

def get_dataset_paths(audio_folder):
    import glob, os, yaml
    #audio_folder = 'MedleyDB_sample/Audio'
    for filename in glob.glob(audio_folder + '/*/*.yaml'):
        metadata = yaml.load(open(filename))
        mix_filename = metadata["mix_filename"]
        vocal_filenames = [
                x['filename'] for x in metadata['stems'].values()
                if 'singer' in x['instrument']]
        if vocal_filenames:
            vocal_path = glob.glob(audio_folder +'/*/*/' + vocal_filenames[0])
            mix_path = glob.glob(audio_folder +'/*/' + mix_filename)
            yield vocal_path[0], mix_path[0]

def get_track_paths(audio_folder):
    import glob, os, yaml
    #audio_folder = 'MedleyDB_sample/Audio'
    for filename in glob.glob(audio_folder + '/*/*.yaml'):
        metadata = yaml.load(open(filename))
        mix_filename = metadata["mix_filename"]
        vocal_filenames = [
                x['filename'] for x in metadata['stems'].values()
                if 'singer' in x['instrument']]
        instruments_filenames = [
                x['filename'] for x in metadata['stems'].values()
                if 'singer' not in x['instrument']]

        if vocal_filenames:
            vocal_paths = []
            instruments_paths = []
            for f in vocal_filenames:
                vocal_paths += glob.glob(audio_folder +'/*/*/' + f)
            for f in instruments_filenames:
                instruments_paths += glob.glob(audio_folder +'/*/*/' + f)

            #mix_path = glob.glob(audio_folder +'/*/' + mix_filename)

            yield vocal_paths, instruments_paths

def mix_wav_files(paths):
    return path

if __name__ == '__main__':
    vocal_paths, instruments_paths = get_track_paths('MedleyDB_sample/Test_Audio')
    vocal_concat_paths = [mix_wav_files(vocal_stems_paths) for vocal_stems_paths in vocal_paths]
    instruments_concat_paths = [mix_wav_files(instruments_stems_paths) for instruments_stems_paths in instruments_paths]
    mix_concat_paths = [mix_wav_files(paths) for paths in zip(vocal_concat_paths, instruments_concat_paths)]

    #prepare_audio_files(zip(vocal_concat_paths, mix_concat_paths))
    prepare_freq_file(vocal_concat_paths, "vocal.stft")
    prepare_freq_file(mix_concat_paths, "mix.stft")
    collect_samples('mix.stft.npy', 'vocal.stft.npy')

    #TODO: prepare test vocal and mix freqs files
    vocal_test_paths = vocal_concat_paths
    mix_test_paths = mix_concat_paths
    prepare_freq_file(vocal_test_paths, "test_vocal.stft")
    prepare_freq_file(mix_test_paths, "test_mix.stft")



    #prepare_audio_files(get_dataset_paths('MedleyDB_sample/Test_Audio'))
