import numpy as np
import os

import config
from core.ssEMnet import ssEMnet
from core.SpatialTransformNetwork import locNet
from core.util import read_array_from_file

'''
Trains a ssEMnet to do 2D affine co-registration 
'''

# Start of user input
toTrain = True
toPredict = True
# End of user input

X1, X2 = read_array_from_file()

net = ssEMnet(X1.shape[1:], X2.shape[1:],
              config.checkpoint_autoencoder_filename, config.checkpoint_ssemnet_filename)
if toTrain:
    net.train(X1, X2)

# Predict?
if toPredict:
    transformed_images = net.predictModel(X1, config.image_sink)
