'''
Trains a ssEMnet to do 2D affine co-registration 
'''

import numpy as np
from scipy.misc import imread

import config
from core.ssEMnet import ssEMnet
from core import util

# 0, id=5 for fixed

moving_images_fns = util.get_file_names(config.training_moving_images_dir)
fixed_images_fns = util.get_file_names('data/mnist_full/training/0', 5, 6)

fixed = util.read_image('data/mnist_full/training/0/56.png')

# X1, X2 = util.split_array(util.get_image_array())

# print(X1.shape)
# print(X1[1:].shape)

for index, m in enumerate(moving_images_fns):
    moving = util.read_image(m)

    net = ssEMnet((config.size, config.size, 1), (config.size, config.size, 1),
                config.checkpoint_autoencoder_filename, config.checkpoint_ssemnet_filename)

    moving_a = np.empty_like([moving])
    moving_a = np.expand_dims(moving_a, axis=3)

    fixed_a = np.empty_like([fixed])
    fixed_a = np.expand_dims(fixed_a, axis=3)

    net.train(fixed_a, moving_a)
    # net.train(moving_a, fixed_a)

    net.predict(fixed_a, moving_a, index, imageSink=config.image_sink)
    # net.predict(moving_a, fixed_a, index, imageSink=config.image_sink)
