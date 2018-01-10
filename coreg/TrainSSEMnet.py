import numpy as np
from ssEMnet import ssEMnet
from MakeDataset import save_array, normalize, load_array
from SpatialTransformNetwork import locNet
import os
import config
from TrainConvAutoencoder import get_file_names, read_images

'''
Trains a ssEMnet to do 2D affine co-registration 
'''

# Start of user input
toTrain = False
toPredict = True
# End of user input

# Prepare data: assumes that X.bc exists already

images_1 = get_file_names(config.image_1_dir)
images_2 = get_file_names(config.image_2_dir)

#TODO: fix paths
if os.path.isfile(config.concatenated_filename):
    X = load_array(config.concatenated_filename)
    num_images = int(X.shape[0] / 2)
    X1 = X[:num_images]
    X2 = X[num_images:]
else:
    print("reloading every image")
    X1 = np.empty((len(images_1), 28, 28))
    X2 = np.empty((len(images_2), 28, 28))
    for i in range(len(images_1)):
        X1[i] = read_images(images_1[i])
    for i in range(len(images_2)):          #TODO: double check if same range length necessary
        X2[i] = read_images(images_2[i])
    X1 = np.expand_dims(X1, axis=3)
    X2 = np.expand_dims(X2, axis=3)
    if not os.path.exists(config.processed_dir):
        os.makedirs(config.processed_dir)
    save_array(config.processed_dir, X1)
    save_array(config.processed_dir, X2)

print("X1 shape: ")
print(X1.shape)

# Train
# Try on very small data set
mynet = ssEMnet(X1.shape[1:], X2.shape[1:], config.checkpoint_autoencoder_filename, config.checkpoint_ssemnet_filename)
if toTrain:
    mynet.train(X1, X2)

# Predict?
if toPredict:
    transformed_images = mynet.predictModel(X1, config.image_sink)
