import os
import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array
import config


# Predict data
X = load_array(config.concatenated_filename)
mynet = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], encoding_decoding_choice=config.encoding_decoding_choice)
mynet.predictModel(X, config.image_sink)
