import os
import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array
import config

checkpoint_filename = os.path.join(config.checkpoint_dir, config.checkpoint_name)

# Predict data
X = load_array(config.processed_dir)
mynet = ConvAutoEncoder2D(
    checkpoint_filename, X.shape[1], X.shape[2], encoding_decoding_choice=config.encoding_decoding_choice)
mynet.predictModel(X, config.image_sink)
