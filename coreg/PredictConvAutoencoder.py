import os
import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array

# User input
ProcessedDirectory = r'data/processed/Coreg_test/X.bc'
ModelFile = r'models/coreg_autoencoder.hdf5'
imageSink = r'data/processed/Coreg_test/check.tif'
# End user input

# Predict data
X = load_array(ProcessedDirectory)
mynet = ConvAutoEncoder2D(ModelFile, X.shape[1], X.shape[2])
mynet.predictModel(X, imageSink)
