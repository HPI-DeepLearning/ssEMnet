from os import path, makedirs
import numpy as np
from skimage import io, transform

from scipy.misc import imread

from core.ConvAutoencoder import ConvAutoEncoder2D
from core import util
import config


X = util.get_image_array()

# create checkpoints folder if needed
if not path.exists(config.checkpoint_dir):
    makedirs(config.checkpoint_dir)

# train
net = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], config.encoding_decoding_choice)
net.train(X)
