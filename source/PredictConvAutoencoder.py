import os
import numpy as np

from .core.ConvAutoencoder import ConvAutoEncoder2D
from .core import util
import config

X = util.get_image_array()

# Predict data
net = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], encoding_decoding_choice=config.encoding_decoding_choice)
net.predictModel(X, config.image_sink)
