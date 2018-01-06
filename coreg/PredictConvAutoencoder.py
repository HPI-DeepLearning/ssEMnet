import os
import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array

import config

concatenated_filename = os.path.join(config.processed_dir, 'concatenated')
checkpoint_filename = os.path.join(config.checkpoint_dir, 'mynet')


# Predict data
X = load_array(concatenated_filename)

mynet = ConvAutoEncoder2D(
    checkpoint_filename, X.shape[1], X.shape[2], encoding_decoding_choice=1)
mynet.predictModel(X)
