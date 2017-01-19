import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model



def get_train_dataset():
    X = pickle.load(open('/home/maori/Documents/mllits/Project/X.pickle', 'rb'))
    #Y = pickle.load(open('/home/maori/Documents/mllits/Project/Y.pickle', 'rb'))
    Y = pickle.load(open('/home/maori/Documents/mllits/Project/mask.pickle', 'rb'))
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def train(X, Y):
    #input_shape = #22051
    model = Sequential()
    model.add(Dense(1025, input_dim=1025, activation='relu'))
    model.add(Dense(1025, activation='sigmoid'))
    # model.compile(optimizer="sgd", loss="mean_squared_logarithmic_error"), mean_absolute_error
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    print("X.shape={}".format(X.shape))
    print("Y.shape={}".format(Y.shape))
    #print(X.shape) = (6135, 1025)
    model.fit(X, Y, nb_epoch=4)
    model.save('my_model.h5')

X, Y = get_train_dataset()
train(X, Y)
