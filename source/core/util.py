from os import path, makedirs, walk

import bcolz
import numpy as np
from skimage import io
from keras import backend as K

import config


def save_array(fname, arr):
    # create directory to save array
    if not path.exists(fname):
        makedirs(fname)

    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def normalize(X):
    X = X.astype('float32')
    return X / (255 * 0.5) - 1

    mean = X.mean()
    std = X.std()

    print('mean: ', mean)
    print('std: ', std)

    print('min before: ', np.amin(X))
    print('max before: ', np.amax(X))

    # Given a set of image data, normalize by subtracting by mean and dividing by the std
    X = X.astype('float32')
    Xn = (X - mean) / std
    print('min: ', np.amin(Xn))
    print('max: ', np.amax(Xn))
    Xn = Xn / np.amax(Xn)
    print(Xn)
    return Xn


def read_image(filename):
    return io.imread(filename, flatten=True, mode='L')


def save_image(filename, image):
    image = K.eval(image)

    image = image[0, :, :, :]
    image = image[:, :, 0]
    io.imsave(filename, image)


def read_array_from_file(filename):
    if path.isdir(filename):
        X = load_array(filename)
        num_images = int(X.shape[0] / 2)
        X1 = X[:num_images]
        X2 = X[num_images:]
        return X1, X2
    else:
        raise Exception(
            'The file for the array did not exists. Mayne it was ever saved?')


def get_file_names(dir, index_from=None, index_to=None):
    for root, dirs, files in walk(dir):
        if index_from is None or index_to is None:
            return [path.join(dir, fn) for fn in files]
        else:
            return [path.join(dir, fn) for fn in files[index_from:index_to]]

def split_array(X):
    num_images = int(X.shape[0] / 2)
    X1 = X[:num_images]
    X2 = X[num_images:]
    return X1, X2

def get_image_array():
    if config.limit:
        moving_images_fns = get_file_names(
            config.training_moving_images_dir, 0, config.limit)
        fixed_images_fns = get_file_names(
            config.training_fixed_images_dir, config.limit, config.limit * 2)
    else:
        moving_images_fns = get_file_names(
            config.training_moving_images_dir)
        fixed_images_fns = get_file_names(
            config.training_fixed_images_dir)

    if False and path.isdir(config.training_saved_filename):
        print('loading saved data')
        return load_array(config.training_saved_filename)
    else:
        moving_images = np.empty((len(moving_images_fns), config.size, config.size))
        for i, fn in enumerate(moving_images_fns):
            moving_images[i] = read_image(fn)

        fixed_images = np.empty(
            (len(fixed_images_fns), config.size, config.size))
        for i, fn in enumerate(fixed_images_fns):
            fixed_images[i] = read_image(fn)

        X = np.concatenate((moving_images, fixed_images), axis=0)
        X = np.expand_dims(X, axis=3)

        save_array(config.training_saved_filename, X)
        return X
