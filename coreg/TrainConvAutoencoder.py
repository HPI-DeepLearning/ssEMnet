import os
import numpy as np
from skimage import io
from pathlib import Path

from chungcore.ConvAutoencoder import ConvAutoEncoder2D
from chungcore.MakeDataset import save_array, normalize, load_array
import config


def get_files(dir):
    for root, dirs, files in os.walk(config.image_1_dir):
        for filename in files:
            print(filename)
        return files


def concatenate_images(image_1_filename, image_2_filename):
    image1 = io.readData(Image1Directory)
    image2 = io.readData(Image2Directory)
    print(image1.shape, image2.shape)
    image = np.concatenate((image1, image2), axis=2)
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 1)
    X = np.expand_dims(image, axis=3)
    save_array(ProcessedDirectory, X)

# image 1
images_1 = get_files(config.image_1_dir)
images_2 = get_files(config.image_2_dir)

for i in range(len(images_1)):


# # Prepare data
# imagenpy = Path(ProcessedDirectory)
# if not os.path.isfile(imagenpy):
#     image1 = io.readData(Image1Directory)
#     image2 = io.readData(Image2Directory)
#     print(image1.shape, image2.shape)
#     image = np.concatenate((image1, image2), axis=2)
#     image = np.swapaxes(image, 1, 2)
#     image = np.swapaxes(image, 0, 1)
#     X = np.expand_dims(image, axis=3)
#     save_array(ProcessedDirectory, X)
# else:
#     X = load_array(ProcessedDirectory)

# # Train
# mynet = ConvAutoEncoder2D(ModelFile, X.shape[1], X.shape[2])
# mynet.train(X)
