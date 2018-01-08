import numpy as np
from ssEMnet import ssEMnet
from MakeDataset import save_array, normalize, load_array
from SpatialTransformNetwork import locNet
import os
import config
from TrainConvAutoencoder import get_file_names, read_images, concatenate_images

'''
Trains a ssEMnet to do 2D affine coregistratoin 
'''

# User input
#ProcessedDirectory = r'D:\analysis\data\processed\coreg_test\X.bc'
ProcessedDirectory = r'data/processed'
ModelFileAutoEncoder = r'models/coreg_autoencoder.hdf5'
ModelFile = r'models/ssemnet.hdf5'
# Where we put predicted transformation images
imageSink = r'results/ssEMnet_crude_2DAffine.tif'
toTrain = False
toPredict = True

# End user input

# Prepare data: assumes that X.bc exists already
#X = load_array(config.processed_dir)

images_1 = get_file_names(config.image_1_dir)
images_2 = get_file_names(config.image_2_dir)

#TODO: fix paths
#if os.path.isfile(config.processed_dir):
#    X1 = load_array(config.processed_dir)
#    X2 = load_array(config.processed_dir)
#else:
X1 = np.empty((len(images_1), 28, 28));
X2 = np.empty((len(images_2), 28, 28));
#X = np.empty((len(images_1), 28, 28))
for i in range(len(images_1)):
    #X[i] = concatenate_images(images_1[i], images_2[i], i)
    X1[i] = read_images(images_1[i])
for i in range(len(images_2)):          #TODO: double check if same range length necessary
    X2[i] = read_images(images_2[i])
X1 = np.expand_dims(X1, axis=3)
X2 = np.expand_dims(X2, axis=3)
if not os.path.exists(config.processed_dir):
    os.makedirs(config.processed_dir)
save_array(config.processed_dir, X1)
save_array(config.processed_dir, X2)


# num_images = int(X.shape[0] / 2)
# X1 = X[:num_images]
# X2 = X[num_images:]
# print("X1 shape: ")
# print(X1.shape)

# Train
# Try on very small data set
mynet = ssEMnet(X1.shape[1:], X2.shape[1:], config.checkpoint_filename, ModelFile)
#mynet = ssEMnet((600, 800, 1), (600, 800, 1), ModelFileAutoEncoder, ModelFile)
if toTrain:
    #mynet.train(X1[:, :600, :800, :], X2[:, :600, :800, :])
    mynet.train(X1, X2)

# Predict?
if toPredict:
    #transformed_images = mynet.predictModel(X1[:, :600, :800, :], imageSink)
    transformed_images = mynet.predictModel(X1, imageSink)
