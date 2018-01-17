import os
import numpy as np
from skimage import io
from pathlib import Path

from core.ConvAutoencoder import ConvAutoEncoder2D
from core.util import save_array, normalize, load_array, get_file_names
import config

# images
images_1_filenames = get_file_names(config.image_1_dir, 0, 100)
images_2_filenames = get_file_names(config.image_2_dir, 100, 200)

print('images1', images_1_filenames)
print('images2', images_2_filenames)

if os.path.isdir(config.concatenated_filename):
    print('loading saved data')
    X = load_array(config.concatenated_filename)
else:
    images_1 = np.empty((len(images_1_filenames), 28, 28))
    for i in range(len(images_1_filenames)):
        images_1[i] = io.imread(images_1_filenames[i])

    images_2 = np.empty((len(images_2_filenames), 28, 28))
    for i in range(len(images_2_filenames)):
        images_2[i] = io.imread(images_2_filenames[i])

    print('shape of images1', images_1.shape)
    X = np.concatenate((images_1, images_2), axis=0)
    print('shape of X', X.shape)
    X = np.expand_dims(X, axis=3)

    # create directory to save array
    if not os.path.exists(config.concatenated_filename):
        os.makedirs(config.concatenated_filename)
    save_array(config.concatenated_filename, X)

print(X.shape)

# create checkpoints folder if needed
if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)

# Train
mynet = ConvAutoEncoder2D(
    config.checkpoint_autoencoder_filename, X.shape[1], X.shape[2], config.encoding_decoding_choice)
mynet.train(X)
