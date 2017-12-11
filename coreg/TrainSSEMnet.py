import numpy as np
from ssEMnet import ssEMnet
from MakeDataset import save_array, normalize, load_array
from SpatialTransformNetwork import locNet
import os

'''
Trains a ssEMnet to do 2D affine coregistratoin 
'''

# User input
ProcessedDirectory = r'D:\analysis\data\processed\Coreg_test\X.bc'
ModelFileAutoEncoder = r'D:\analysis\models\coreg_autoencoder.hdf5'
ModelFile = r'D:\analysis\models\ssemnet.hdf5'
# Where we put predicted transformation images
imageSink = r'D:\analysis\results\coreg_test\ssEMnet_crude_2DAffine.tif'
toTrain = False
toPredict = True
# End user input

# Prepare data: assumes that X.bc exists already
X = load_array(ProcessedDirectory)
num_images = int(X.shape[0] / 2)
X1 = X[:num_images]
X2 = X[num_images:]

# Train
# Try on very small data set
# mynet = ssEMnet(X1.shape[1:], X2.shape[1:], ModelFileAutoEncoder, ModelFile)
mynet = ssEMnet((600, 800, 1), (600, 800, 1), ModelFileAutoEncoder, ModelFile)
if toTrain:
    mynet.train(X1[:, :600, :800, :], X2[:, :600, :800, :])

# Predict?
if toPredict:
    transformed_images = mynet.predictModel(X1[:, :600, :800, :], imageSink)
