import glob
import os
import bcolz
import numpy as np


def save_array(fname, arr): c = bcolz.carray(
    arr, rootdir=fname, mode='w'); c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def normalize(X):
    # Given a set of image data, normalize by subtracting by mean and dividing by the std
    X = X.astype('float32')
    Xn = (X - X.mean()) / X.std()
    Xn = Xn / np.amax(Xn)
    return Xn
