import numpy as np
import clarity.IO as io
from pathlib import Path
from ConvAutoencoder import ConvAutoEncoder2D
from MakeDataset import save_array, normalize, load_array
import os

# User input
Image1Directory = r'D:\analysis\data\raw\Coreg_test\R1.tif'
Image2Directory = r'D:\analysis\data\raw\Coreg_test\R2.tif'
ProcessedDirectory = r'D:\analysis\data\processed\Coreg_test\X.bc'
ModelFile = r'D:\analysis\models\coreg_autoencoder.hdf5'
# End user input

# Prepare data
imagenpy = Path(ProcessedDirectory)
if not os.path.isfile(imagenpy):
    image1 = io.readData(Image1Directory)
    image2 = io.readData(Image2Directory)
    print(image1.shape, image2.shape)
    image = np.concatenate((image1, image2), axis=2)
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 1)
    X = np.expand_dims(image, axis=3)
    save_array(ProcessedDirectory, X)
else:
    X = load_array(ProcessedDirectory)

# Train
mynet = ConvAutoEncoder2D(ModelFile, X.shape[1], X.shape[2])
mynet.train(X)
