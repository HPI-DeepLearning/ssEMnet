import copy
import os
import glob
import bcolz
import numpy as np
from skimage import io
from keras import backend as K

import config

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


def save_image(filename, image):
    # tf_session = K.get_session()

    image = K.eval(image)

    print('before', image.shape)
    image = image[0, :, :, :]
    image = image[:, :, 0]
    print(image)
    print('after', image.shape)
    io.imsave(filename, image)

def read_array_from_file():
    if os.path.isdir(config.concatenated_filename):
        X = load_array(config.concatenated_filename)
        num_images = int(X.shape[0] / 2)
        X1 = X[:num_images]
        X2 = X[num_images:]
        return X1, X2
    else:
        raise Exception(
            'You have to run TrainConvAutoencoder to create the data.')


def get_file_names(dir, index_from, index_to):
    for root, dirs, files in os.walk(dir):
        if index_from is None or index_to is None:
            return [os.path.join(dir, fn) for fn in files]
        else:
            return [os.path.join(dir, fn) for fn in files[index_from:index_to]]
