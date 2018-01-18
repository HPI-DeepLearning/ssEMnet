import config
from core.ssEMnet import ssEMnet
from core.util import read_array_from_file, get_file_names

from scipy.misc import imread

import numpy as np
'''
Trains a ssEMnet to do 2D affine co-registration 
'''

# X1, X2 = read_array_from_file(config.training_saved_filename)

moving_images_fns = get_file_names('data/raw/mnist_png/training/1', 0, 1)
fixed_images_fns = get_file_names('data/raw/mnist_png/training/1', 1, 2)

moving_images = np.empty((len(moving_images_fns), 28, 28))
for i in range(len(moving_images_fns)):
    img = imread(moving_images_fns[i], flatten=True, mode='L')
    moving_images[i] = img

fixed_images = np.empty((len(fixed_images_fns), 28, 28))
for i in range(len(fixed_images_fns)):
    img = imread(fixed_images_fns[i], flatten=True, mode='L')
    fixed_images[i] = img

moving_images = np.expand_dims(moving_images, axis=3)
fixed_images = np.expand_dims(fixed_images, axis=3)

net = ssEMnet(moving_images.shape[1:], fixed_images.shape[1:],
              config.checkpoint_autoencoder_filename, config.checkpoint_ssemnet_filename)

net.train(moving_images, fixed_images)

net.predict(moving_images, fixed_images, imageSink=config.image_sink)
