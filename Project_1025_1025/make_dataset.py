#!/usr/bin/env python
import glob
import os
from shutil import copyfile
import argparse

import librosa
import sox
import yaml
import numpy as np


def prepare_freq_file(paths, freq_file_path):
    """Extract frequencies from audio files
    using STFT (short time Fourier transformation).
    Save resulting frequencies to the file.
    
    Parameters
    ----------
    paths : list
        The list of audio files paths.
    freq_file_path : string
        Path of the file for saving frequencies gotten by stft.

    """
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
    """Makes X and Y samples and saves them to X.npy and Y.npy. 
    
    Parameters
    ----------
    x_file_name : string
        Stft data file path of mix audio. 
    y_file_name : string
        Stft data file path of vocal audio.
    """
    print("Load resampled stft files")
    x = np.load(x_file_name)
    y = np.load(y_file_name)
    assert len(x) == len(y), "Mix,Vocal and Instrument lengths don't match"

    print("Pick samples")
    collected_samples = 0
    rng = np.random.RandomState(0)
    X = []
    Y = []
    examples_wanted = len(x) #1000
    while collected_samples < examples_wanted:
        pos = rng.randint(len(x))
        sample_x = x[pos]
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

    
def get_track_paths(audio_folder):
    """ Get paths to vocal and instruments files of tracks.
    
    Parameters
    ----------
    audio_folder : string
        Path to audio tracks folder (MedleyDB/Audio)
    
    Yields
    -------
    vocal_paths, instruments_paths : vocal and instruments files \
                            from all track folders in audio_folder.
    """
    res_filenames = []
    for filename in glob.glob(audio_folder + '/*/*.yaml'):
        print(filename)
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

        if vocal_paths and instruments_paths:
            result = vocal_paths, instruments_paths
            yield vocal_paths, instruments_paths


def glue_wav_files(paths, glue_wav_path):
    """Glue wav audio files (from paths) to one audio file(glue_wav_path).
    
    Parameters
    ----------
    paths : list
        List of audio filenames
    glue_wav_paths: string
        Filename of resulting wav file.
    """
    if not paths:
        return
    if len(paths) == 1:
        copyfile(paths[0], glue_wav_path)
    else:
        combiner = sox.Combiner()
        combiner.build(paths, glue_wav_path, combine_type = 'mix-power')
        
        
def make_vocal_instrum_mix_files(track_path):
    """Make vocal, instruments, mix wav files in track path.
    
    Parameters
    ----------
    track_path : string
        Path to track files. 
    """
    if track_path == ([], []):
        return
    print(track_path)
    vocal_paths, instrum_paths = track_path
    vocal_concat_path = os.path.join(os.path.dirname(vocal_paths[0]), \
                                                 "..", "..", vocal.wav")
    glue_wav_files(vocal_paths, vocal_concat_path)
    instrum_concat_path = os.path.join(os.path.dirname(instrum_paths[0]), \
                                               "..", "..", "instrument.wav")
    glue_wav_files(instrum_paths, instrum_concat_path)
    mix_concat_path = os.path.join(os.path.dirname(vocal_concat_path), "..", "mix.wav")
    glue_wav_files([vocal_concat_path, instrum_concat_path], mix_concat_path)

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-audio-path', 
                        default='../MedleyDB_sample/Train_Audio',
                        help="Path to train audio path, default is %(default)s"
                        )
    parser.add_argument('--valid-audio-path', 
                        default='../MedleyDB_sample/Valid_Audio',
                        help="Path to validation audio path, default is %(default)s"
                        )
    args = parser.parse_args()
    
    paths = {'train':args.train_audio_path, 'valid':args.valid_audio_path}
    # Create vocal, instruments and mix wav files.
    for path in paths.values():
        for track_path in get_track_paths(path):
            make_vocal_instrum_mix_files(track_path)

    #Prepare train and valid vocal and mix freqs files
    for prefix, audio_path in paths.items():
        vocal_paths = glob.glob(os.path.join(audio_path, '*', 'vocal.wav'))
        mix_paths = glob.glob(os.path.join(audio_path, '*', 'mix.wav'))
        instrument_paths = glob.glob(os.path.join(audio_path, '*', 'instrument.wav'))
        prepare_freq_file(vocal_paths, prefix + "_vocal.stft")
        prepare_freq_file(mix_paths, prefix + "_mix.stft")
    
