from skimage import io
import numpy as np
from keras import backend as K
import copy

def save_image(filename, image):
    # tf_session = K.get_session()

    image = K.eval(image)

    print('before', image.shape)
    image = image[0, :, :, :]
    image = image[:, :, 0]
    print(image)
    print('after', image.shape)
    io.imsave(filename, image)
