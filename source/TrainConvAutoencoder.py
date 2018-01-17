from os import path, makedirs
import numpy as np
from skimage import io

from core.ConvAutoencoder import ConvAutoEncoder2D
from core.util import save_array, normalize, load_array, get_file_names
import config

# images
moving_images_fns = get_file_names(config.training_moving_images_dir)
fixed_images_fns = get_file_names(config.training_fixed_images_dir)

if path.isdir(config.training_saved_filename):
    print('loading saved data')
    X = load_array(config.training_saved_filename)
else:
    moving_images = np.empty((len(moving_images_fns), 28, 28))
    for i in range(len(moving_images_fns)):
        moving_images[i] = io.imread(moving_images_fns[i])

    fixed_images = np.empty((len(fixed_images_fns), 28, 28))
    for i in range(len(fixed_images_fns)):
        fixed_images[i] = io.imread(fixed_images_fns[i])

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
