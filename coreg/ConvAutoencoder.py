import numpy as np
from MakeDataset import normalize
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Reshape, merge
from keras.layers import Activation, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy, binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras import regularizers
from skimage import io

import util

'''
Module that introduces a convolutional autoencoder
'''


class ConvAutoEncoder2D(object):
    def __init__(self, ModelFile, img_rows, img_cols, encoding_decoding_choice=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.ModelFile = ModelFile
        self.encoding_decoding_choice = encoding_decoding_choice

    def encode(self, inputs, layer_start_index):
        # layer_starT_index is so we can identify the layers by name in ssSEMnet: input is the starting index for which we wish to name
        # e.g. layer_start_index = 1 --> the layers will be named encode1, encode2, ...etc
        indices = list(
            map(str, list(range(layer_start_index, layer_start_index + 6))))
        names = []
        for element in indices:
            names.append('encode_' + element)
        encoded = Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[0])(inputs)
        encoded = MaxPooling2D((2, 2), name=names[1])(encoded)
        encoded = Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[2])(encoded)
        encoded = MaxPooling2D((2, 2), name=names[3])(encoded)
        encoded = Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[4])(encoded)
        encoded = MaxPooling2D((2, 2), name=names[5])(encoded)
        return encoded

    def decode(self, encoded_input):
        indices = list(map(str, list(range(7))))
        names = []
        for element in indices:
            names.append('decode_' + element)

        decoded = Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[0])(encoded_input)
        decoded = UpSampling2D((2, 2), name=names[1])(decoded)
        decoded = Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[2])(decoded)
        decoded = UpSampling2D((2, 2), name=names[3])(decoded)
        decoded = Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[4])(decoded)
        decoded = UpSampling2D((2, 2), name=names[5])(decoded)
        decoded = Conv2D(1, (3, 3), activation='tanh', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[6])(decoded)
        return decoded


    def encode_2(self, inputs, layer_start_index):
         # layer_starT_index is so we can identify the layers by name in ssSEMnet: input is the starting index for which we wish to name
        # e.g. layer_start_index = 1 --> the layers will be named encode1, encode2, ...etc
        indices = list(
            map(str, list(range(layer_start_index, layer_start_index + 2))))
        names = []
        for element in indices:
            names.append('encode_' + element)

        encoded = Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[0])(inputs)
        encoded = MaxPooling2D((2, 2), name=names[1])(encoded)
        return encoded

    def decode_2(self, encoded_input):
        indices = list(map(str, list(range(3))))
        names = []
        for element in indices:
            names.append('decode_' + element)

        decoded = Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[0])(encoded_input)
        decoded = UpSampling2D((2, 2), name=names[1])(decoded)
        decoded = Conv2D(1, (3, 3), activation='tanh', padding='same',
                         kernel_regularizer=regularizers.l2(1e-3), name=names[2])(decoded)
        # util.save_image('test', decoded)
        return decoded

    def getAutoencoder(self, inputs):
        if self.encoding_decoding_choice is None:
            encoded = self.encode(inputs, 1)
            decoded = self.decode(encoded)
        else:
            encoded = self.encode_2(inputs, 1)
            decoded = self.decode_2(encoded)

        print('getA', decoded)
        model = Model(inputs, decoded)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        print('getA', decoded)
        return model

    def train(self, X):
        # X is a 4D array of the images
        X = normalize(X)
        inputs = Input((self.img_rows, self.img_cols, 1))
        model = self.getAutoencoder(inputs)
        model_checkpoint = ModelCheckpoint(
            self.ModelFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        # X is the input and also the output
        model.fit(X, X, batch_size=1, epochs=10, shuffle=True, verbose=1,
                  validation_split=0.2, callbacks=[model_checkpoint])

    def predictModel(self, datas, imageSink=None):
        inputs = Input((self.img_rows, self.img_cols, 1))
        model = self.getAutoencoder(inputs)
        model.load_weights(self.ModelFile)

        print('predict some data')
        datas = normalize(datas)
        imgs_mask_test = model.predict(datas, batch_size=1, verbose=1)
        print(imgs_mask_test.shape)
        imgs_mask_test = np.swapaxes(imgs_mask_test, 0, 1)
        imgs_mask_test = np.swapaxes(imgs_mask_test, 1, 2)

        # Since all the pixel values are between -1 and 1, map to be between 0 and 1.
        imgs_mask_test = (imgs_mask_test / np.amax(imgs_mask_test) + 1) * 0.5
        imgs_mask_test = np.swapaxes(imgs_mask_test, 2, 1)
        imgs_mask_test = np.swapaxes(imgs_mask_test, 1, 0)
        if not imageSink is None:

            for i in range(imgs_mask_test.shape[0]):
                image = imgs_mask_test[i][:, :, -1]
                io.imsave(imageSink + "autoencoderResult" + str(i) + ".png", image)

        return imgs_mask_test
