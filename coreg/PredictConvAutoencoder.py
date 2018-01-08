import os
import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array
import config

# User input
#ProcessedDirectory = r'data/processed/X.bc'
ProcessedDirectory = r'data/processed'

ModelFile = r'models/coreg_autoencoder.hdf5'
imageSink = r'data/processed/check.tif'
#imageSink = r'results/ssEMnet_crude_2DAffine.tif'
# End user input

# Predict data
X = load_array(ProcessedDirectory)
mynet = ConvAutoEncoder2D(ModelFile, X.shape[1], X.shape[2], encoding_decoding_choice=config.encoding_decoding_choice)
mynet.predictModel(X, imageSink)
