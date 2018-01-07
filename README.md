# Music-Voice Separation

The goal of this project is to extract singing voice from music audio records. This is done by a neural network that learns which frequences from audio-mix belong to vocals and which to instruments.


## Install

Install libsox::

    $ sudo apt-get install libsox-dev

Next, install dependencies::

    $ pip install -r requirements.txt

## Dataset

We train on MedleyDB, a dataset of annotated multitrack audio recordings.
It consists of 122 multitrack songs.

The dataset (96 GB of data) is available by request at http://medleydb.weebly.com/


## Workflow

### 1. Prepare data

...

### 2. Train model

```
$ ./train.py 
```

### 3. Split voice and instruments:

```
$ ./predict.py path/to/music.wav
```
