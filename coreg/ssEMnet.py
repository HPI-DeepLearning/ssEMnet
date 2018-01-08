import numpy as np
from ConvAutoencoder import ConvAutoEncoder2D
from SpatialTransformNetwork import *
from objectives import generic_unsupervised_loss
from MakeDataset import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage import io

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

        model = Model((input1, input2), z)

        # Placeholder for autoencoder so we can load the trained weights into our encoder
        # encode layers start naming at 1 and end at 6
        autoencoder = Model(input1, g.decode_2(g.encode_2(input1, 1)))
        #model.summary() # for debug
        autoencoder.load_weights(self.ModelFileAutoEncoder)

        # Now make weights untrainable in the encoder level
        for l in model.layers:
            if 'encode' in l.name:
                #print(l.name)
                #i = int(l.name[6:])  # the number of the encode layer
                i = int(l.name[7:])  # the number of the encode layer
                #print(i)
                autoencoder.summary()
                weights = autoencoder.get_layer(
                    'encode_' + str((i + 5) % 6 + 1)).get_weights()
                l.set_weights(weights)
                l.trainable = False  # Make encoder layers not trainable

        model.compile(optimizer=Adam(), loss=generic_unsupervised_loss)
        model.summary()
        return model

    def getPredictNet(self):
        # Calculate the transformation of the moving images to the fixed images

        input1 = Input(self.input1_shape)
        x = SpatialTransformer(localization_net=self.locnet, output_size=self.input1_shape,
                               input_shape=self.input1_shape, name='stm')(input1)
        model = Model(input1, x)
        model.summary()
        return model

    def train(self, X1, X2):
        X1 = normalize(X1)
        X2 = normalize(X2)
        model = self.getssEMnet()
        model_checkpoint = ModelCheckpoint(
            self.ModelFile, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        model.fit([X1, X2], np.zeros((X1.shape[0],)), batch_size=1, epochs=10,
                  shuffle=True, verbose=1, validation_split=0.2, callbacks=[model_checkpoint])

    def predictModel(self, X1, imageSink):
        # Load weights of just the spatial transformer network
        print("available images to predict: ", X1.shape)
        model = self.getssEMnet()
        model.load_weights(self.ModelFile)

        predicted = self.getPredictNet()
        # Load the appropriate weights into the spatial transformer layer
        stm_weights = model.get_layer('stm').get_weights()

        #print('stm_weights: ', stm_weights)
        print('stm_weights length: ', len(stm_weights))
        predicted.get_layer('stm').set_weights(stm_weights)

        X1 = normalize(X1)
        transformed_images = predicted.predict(X1, batch_size=1, verbose=1)
        #print('transformed_images: ', transformed_images)
        print('transformed_images length: ', len(transformed_images))
        transformed_images = transformed_images / \
            np.amax(abs(transformed_images))
        transformed_images = (transformed_images + 1) * 0.5
        transformed_images = np.swapaxes(transformed_images, 0, 1)
        transformed_images = np.swapaxes(transformed_images, 1, 2)

        # This is using clarity.IO
        if not imageSink is None:
            #io.writeData(imageSink, transformed_images)

            # Open a file
            #fo = open(imageSink, "w")
            #fo.write(transformed_images)

            # Close opend file
            #fo.close()

            #np.save(imageSink, transformed_images)
            io.imsave(imageSink+"bla.jpeg", transformed_images)

        return transformed_images
