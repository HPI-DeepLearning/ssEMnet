import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage import io

from .ConvAutoencoder import ConvAutoEncoder2D
from .SpatialTransformNetwork import *
from .objectives import generic_unsupervised_loss
from .util import *


'''
Module that tries to replicate ssEMnet (use weights from trained autoencoder)
and input into a spatial transformation network - done in 2D
'''


def mse_image_similarity(images):
    # Function that calculates the mean squared difference between two images
    image1 = images[0]
    image2 = images[1]
    img1 = K.flatten(image1)
    img2 = K.flatten(image2)
    return K.mean(K.square(img1 - img2))


class ssEMnet(object):
    def __init__(self, input1_shape, input2_shape, ModelFileAutoEncoder, ModelFile):
        self.ModelFile = ModelFile
        self.ModelFileAutoEncoder = ModelFileAutoEncoder
        self.input1_shape = input1_shape
        self.input2_shape = input2_shape
        self.locnet = locNet(self.input1_shape)

    def getEncoder(self, inputs):
        # Get placeholder for original autoencoder model so we can transfer weights to encoder

        # Assumes that we have already trained an autoencoder and that we would like to load old weights in.
        encoded = self.encode(inputs)
        model = Model(inputs, encoded)
        i = 0  # index to keep track of which layer we are on.
        for l in model.layers:
            weights = autoencoder.layers[i].get_weights()
            l.trainable = False  # Make layer untraniable: just train spatial transformer network
            l.set_weights(weights)
            i += 1

        # model.summary()
        return model

    def getssEMnet(self):
        # input1_shape = moving image (that which we are transforming)
        # input2_shape = fixed image (that which we are targeting to)

        input1 = Input(self.input1_shape)
        input2 = Input(self.input2_shape)
        x = SpatialTransformer(localization_net=self.locnet, output_size=self.input1_shape,
                               input_shape=self.input1_shape, name='stm')(input1)
        g = ConvAutoEncoder2D(self.ModelFileAutoEncoder, x.get_shape().as_list()[
                              0], x.get_shape().as_list()[1])
        x1 = g.encode_2(x, 1)

        # Produce feature map of target images
        y = g.encode_2(input2, 7)

        # Now we want to calculate the loss, we measure similarity between images. Should be scalar
        z = Lambda(mse_image_similarity)([x1, y])

        model = Model((input1, input2), (z, x))

        # Placeholder for autoencoder so we can load the trained weights into our encoder
        # encode layers start naming at 1 and end at 6
        autoencoder = Model(input1, g.decode_2(g.encode_2(input1, 1)))
        #model.summary() # for debug
        autoencoder.load_weights(self.ModelFileAutoEncoder)

        # Now make weights untrainable in the encoder level
        for l in model.layers:
            if 'encode' in l.name:
                i = int(l.name[7:]) # cut of chars to get the number of the encode layer
                print('Encode Index')
                print(i)

                index_in_autoencoder = (i + 5) % 6 + 1
                print(index_in_autoencoder)

                autoencoder.summary()
                weights = autoencoder.get_layer(
                    'encode_' + str(index_in_autoencoder)).get_weights()
                l.set_weights(weights)
                l.trainable = False  # Make encoder layers not trainable

        model.compile(optimizer=SGD(lr=0.3),
                      loss=generic_unsupervised_loss,
                      loss_weights=[1., 0.0]) # ignores the loss of transformed image (the 2. output)
        model.summary()
        return model

    def train(self, X1, X2):
        X1 = normalize(X1)
        X2 = normalize(X2)
        model = self.getssEMnet()

        model_checkpoint = ModelCheckpoint(
            self.ModelFile, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True)
        model.fit([X1, X2],
            [np.zeros((X1.shape[0],)), np.zeros(X1.shape,)], # the first one is the distance between the images to optimise, the second the transformed image (ignored for loss)
            batch_size=1,
            epochs=50,
            shuffle=False, verbose=1, validation_split=0.2, callbacks=[model_checkpoint])

    def predict(self, X1, X2, imageSink):
        X1 = normalize(X1)
        X2 = normalize(X2)
        model = self.getssEMnet()
        [_, results] = model.predict([X1, X2], batch_size=1, verbose=1)
        results = results / \
            np.amax(abs(results))
        results = (results + 1) * 0.5
        if not imageSink is None:
            for i in range(results.shape[0]):
                image = results[i][:, :, -1]
                io.imsave(imageSink + "ssemnet_transformed_joe" +
                          str(i) + ".png", image)
        return results
