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

if os.path.isdir(config.concatenated_filename):
    X = load_array(config.concatenated_filename)
    num_images = int(X.shape[0] / 2)
    X1 = X[:num_images]
    X2 = X[num_images:]
else:
    raise Exception('You have to run TrainConvAutoencoder to create the data.')

net = ssEMnet(X1.shape[1:], X2.shape[1:],
                config.checkpoint_autoencoder_filename, config.checkpoint_ssemnet_filename)
if toTrain:
    net.train(X1, X2)

# Predict?
if toPredict:
    transformed_images = net.predictModel(X1, config.image_sink)
