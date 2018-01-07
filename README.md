# Music-Voice Separation

The goal of this project is to extract singing voice from music audio records.
This is done by a neural network that learns which frequences from audio-mix
belong to vocals and which to instruments.


## Install

Install libsox::

    $ sudo apt-get install libsox-dev

Next, install dependencies::

    $ pip install -r requirements.txt


## Dataset

We train on MedleyDB, a dataset of annotated multitrack audio recordings.
It consists of 122 multitrack songs.

The dataset (96 GB of data) is available by request at http://medleydb.weebly.com/

`preprocess.py` script will make the following transformations and extract features:

1. Downsample audio to 16,000 samples/sec.

2. Move to frequencies domain by doing Short-Time Fourier Transformations.

3. Phase component is ignored.

4. Input features are target sample with some context (samples to the left and
   to the right from the target).


## Workflow

### 1. Prepare data

Having MedleyDB downloaded and unpacked in the project's root folder, run

```
$ ./preprocess.py
```

### 2. Train model

```
$ ./model.py train --epochs 30
```

### 3. Split voice and instruments:

```
$ ./model.py predict path/to/mix.wav output.wav
```
