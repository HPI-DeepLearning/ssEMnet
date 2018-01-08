import os
import numpy as np
from skimage import io
from pathlib import Path
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array
import config

def get_files(dir):
    for root, dirs, files in os.walk(config.image_1_dir):
        return [os.path.join(config.image_1_dir, fn) for fn in files]


def concatenate_images(image_1_filename, image_2_filename, i):
    image1 = io.imread(image_1_filename)
    image2 = io.imread(image_2_filename)
    print(image1.shape, image2.shape)
    image = np.concatenate((image1, image2), axis=0)
    # image = np.concatenate((image1, image2), axis=2)
    # image = np.swapaxes(image, 1, 2)
    # image = np.swapaxes(image, 0, 1)
    # X = np.expand_dims(image, axis=3)
    return image

def read_images(image_filename):
    print(image_filename)
    return io.imread(image_filename)


# image 1
images_1 = get_files(config.image_1_dir)
images_2 = get_files(config.image_2_dir)

print('images1', images_1)
print('images2', images_2)

#concatenated_filename = os.path.join(config.processed_dir, 'concatenated')

if os.path.isfile(config.concatenated_filename):
    X = load_array(config.concatenated_filename)
else:
    #X = np.empty((len(images_1), 28*2, 28));
    X = np.empty((len(images_1), 28, 28))
    for i in range(len(images_1)):
        #X[i] = concatenate_images(images_1[i], images_2[i], i)
        X[i] = read_images(images_1[i])
    X = np.expand_dims(X, axis=3)
    if not os.path.exists(config.concatenated_filename):
        os.makedirs(config.concatenated_filename)
    save_array(config.concatenated_filename, X)

print(X.shape)

# create checkpoints folder if needed
if not os.path.exists(config.checkpoint_dir):
    os.makedirs(config.checkpoint_dir)

# Train
mynet = ConvAutoEncoder2D(
    config.checkpoint_filename, X.shape[1], X.shape[2], config.encoding_decoding_choice)
mynet.train(X)


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
