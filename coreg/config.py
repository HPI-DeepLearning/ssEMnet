import os

image_1_dir = 'data/mnist_min/0_1'
image_2_dir = 'data/mnist_min/0_2'
processed_dir = 'data/processed'
checkpoint_dir = 'checkpoints'
checkpoint_autoencoder_name = 'autoencoderModel.hdf5'
checkpoint_ssemnet_name = 'ssemnetModel.hdf5'
checkpoint_autoencoder_filename = os.path.join(
    checkpoint_dir, checkpoint_autoencoder_name)
checkpoint_ssemnet_filename = os.path.join(
    checkpoint_dir, checkpoint_ssemnet_name)
encoding_decoding_choice = 1
image_sink = 'results/'
concatenated_filename = os.path.join(processed_dir, 'concatenated.bc')
