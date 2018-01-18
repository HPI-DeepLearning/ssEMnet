from os import path, makedirs
import numpy as np
from skimage import io, transform

from scipy.misc import imread

from core.ConvAutoencoder import ConvAutoEncoder2D
from core.util import save_array, normalize, load_array, get_file_names
import config

import scipy.ndimage as ndimage

# # images
# moving_images_fns = get_file_names(config.training_moving_images_dir)
# fixed_images_fns = get_file_names(config.training_fixed_images_dir)

moving_images_fns = get_file_names('data/raw/mnist_png/training/1', 0, 200)
fixed_images_fns = get_file_names('data/raw/mnist_png/training/1', 200, 400)

if config.encoding_decoding_choice is None:
    size = 128
else:
    size = 28

if False and path.isdir(config.training_saved_filename):
    print('loading saved data')
    X = load_array(config.training_saved_filename)
else:
    moving_images = np.empty((len(moving_images_fns), size, size))
    for i in range(len(moving_images_fns)):
        img = imread(moving_images_fns[i], flatten=True, mode='L')

        # moving_images[i] = transform.resize(img, (size, size))
        moving_images[i] = img

    fixed_images = np.empty((len(fixed_images_fns), size, size))
    for i in range(len(fixed_images_fns)):
        img = imread(fixed_images_fns[i], flatten=True, mode='L')

        # fixed_images[i] = transform.resize(img, (size, size))
        fixed_images[i] = img


    X = np.concatenate((moving_images, fixed_images), axis=0)
    X = np.expand_dims(X, axis=3)

    save_array(config.training_saved_filename, X)

# create checkpoints folder if needed
if not path.exists(config.checkpoint_dir):
    makedirs(config.checkpoint_dir)

# train
net = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], config.encoding_decoding_choice)
net.train(X)
