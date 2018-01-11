import os

mnist_image_1_dir = 'data/mnist_min/0_1'
mnist_image_2_dir = 'data/mnist_min/0_2'
image_1_dir = 'data/cardiac/ES_rescaled' #moving
image_2_dir = 'data/cardiac/ED_rescaled' #fixed
processed_dir = 'data/processed'
checkpoint_dir = 'checkpoints'
checkpoint_autoencoder_name = 'autoencoderModel.hdf5'
checkpoint_ssemnet_name = 'ssemnetModel.hdf5'
checkpoint_autoencoder_filename = os.path.join(checkpoint_dir, checkpoint_autoencoder_name)
checkpoint_ssemnet_filename = os.path.join(checkpoint_dir, checkpoint_ssemnet_name)
encoding_decoding_choice = 1
image_sink = 'results/'
concatenated_filename = os.path.join(processed_dir, 'concatenated')
image_height = 246
image_width = 222