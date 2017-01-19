import librosa
import numpy as np
import sox
from shutil import copyfile
import glob

def prepare_freq_file(paths, freq_file_path):
        target_rate = 16384
        print("Resample files")
        files_freqs = []
        for f in paths:
            sample, rate = librosa.load(f, sr=target_rate)
            print("STFT")
            freqs = np.stack(librosa.core.stft(sample), axis=1) # (..., 1025)

            if files_freqs != []:
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
    examples_wanted = 1000 #len(x)
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
    res_filenames = []
    for filename in glob.glob(audio_folder + '/*/*.yaml'):
        metadata = yaml.load(open(filename))
        mix_filename = metadata["mix_filename"]
        print()
        vocal_filenames = [
                x['filename'] for x in metadata['stems'].values()
                if 'singer' in x['instrument'] or 'vocalists' in x['instrument']]
        instruments_filenames = [
                x['filename'] for x in metadata['stems'].values()
                if 'singer' not in x['instrument'] and 'vocalists' not in x['instrument']]

        vocal_paths = []
        instruments_paths = []
        for f in vocal_filenames:
            vocal_paths += glob.glob(audio_folder +'/*/*/' + f)
        for f in instruments_filenames:
            instruments_paths += glob.glob(audio_folder +'/*/*/' + f)

        result = ()
        if vocal_paths == [] or instruments_paths == []:
            result = [], []
        else:
            result = vocal_paths, instruments_paths

        res_filenames.append(result)

    return res_filenames

def mix_wav_files(paths, mix_wav_path):
    if not paths:
        return ''
    if len(paths) == 1:
        copyfile(paths[0], mix_wav_path)
    else:
        combiner = sox.Combiner()
        combiner.build(paths, mix_wav_path, combine_type = 'mix-power')

if __name__ == '__main__':
    audio_path = 'MedleyDB/Audio'

    #track_paths = get_track_paths(audio_path)
    for track_path in []:#track_paths:
        if track_path != ([], []):
            vocal_paths, instrument_paths = track_path

            vocal_concat_path = '/'.join(vocal_paths[0].split("/")[:-2]) + "/vocal.wav"
            mix_wav_files(vocal_paths, vocal_concat_path)

            instrument_concat_path = '/'.join(instrument_paths[0].split("/")[:-2]) + "/instrument.wav"
            mix_wav_files(instrument_paths, instrument_concat_path)

            mix_concat_path = '/'.join(vocal_concat_path.split("/")[:-1]) + "/mix.wav"
            mix_wav_files([vocal_concat_path, instrument_concat_path], mix_concat_path)


    if False:
        train_audio_path = 'MedleyDB_sample/Train_Audio'
        train_vocal_paths = glob.glob(train_audio_path +'/*/vocal.wav')
        train_mix_paths = glob.glob(train_audio_path +'/*/mix.wav')
        prepare_freq_file(train_vocal_paths, "train_vocal.stft")
        prepare_freq_file(train_mix_paths, "train_mix.stft")
        collect_samples('train_mix.stft.npy', 'train_vocal.stft.npy')

    #TODO: prepare test vocal and mix freqs files
    test_audio_path = 'MedleyDB_sample/Test_Audio'
    test_vocal_paths = glob.glob(test_audio_path +'/*/vocal.wav')
    test_mix_paths = glob.glob(test_audio_path +'/*/mix.wav')
    test_instrument_paths = glob.glob(test_audio_path +'/*/instrument.wav')

    prepare_freq_file(test_vocal_paths, "test_vocal.stft")
    prepare_freq_file(test_mix_paths, "test_mix.stft")



    #prepare_audio_files(get_dataset_paths('MedleyDB_sample/Test_Audio'))
