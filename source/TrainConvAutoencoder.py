from os import path, makedirs
import numpy as np
from skimage import io

from scipy.misc import imread

from core.ConvAutoencoder import ConvAutoEncoder2D
from core.util import save_array, normalize, load_array, get_file_names
import config

import scipy.ndimage as ndimage

# images
moving_images_fns = get_file_names(config.training_moving_images_dir)
fixed_images_fns = get_file_names(config.training_fixed_images_dir)

if False and path.isdir(config.training_saved_filename):
    print('loading saved data')
    X = load_array(config.training_saved_filename)
else:
    moving_images = np.empty((len(moving_images_fns), 28, 28))
    for i in range(len(moving_images_fns)):
        moving_images[i] = imread(
            moving_images_fns[i], flatten=True, mode='L')
        # moving_images[i] = io.imread(moving_images_fns[i], as_grey=True)

    fixed_images = np.empty((len(fixed_images_fns), 28, 28))
    for i in range(len(fixed_images_fns)):
        fixed_images[i] = imread(fixed_images_fns[i], flatten=True, mode='L')
        # fixed_images[i] = io.imread(fixed_images_fns[i], as_grey=True)


    X = np.concatenate((moving_images, fixed_images), axis=0)
    X = np.expand_dims(X, axis=3)

    # print(moving_images[0])
    # print(moving_images[0].shape)

    # io.imsave('fuck.png', moving_images[0])

    save_array(config.training_saved_filename, X)

# create checkpoints folder if needed
if not path.exists(config.checkpoint_dir):
    makedirs(config.checkpoint_dir)

# train
net = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], config.encoding_decoding_choice)
net.train(X)
