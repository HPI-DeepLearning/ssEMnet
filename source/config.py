from os import path

dataset_name = 'mnist_full'
size = 28

limit = 200

training_fixed_images_dir = path.join(
    'data', dataset_name, 'training', '0')
training_moving_images_dir = path.join(
    'data', dataset_name, 'training', '0')

saved_dir = path.join('data', 'saved',)
training_saved_filename = path.join(
    saved_dir, dataset_name + 'training_processed.bc')

checkpoint_dir = 'checkpoints'
checkpoint_autoencoder_name = '_autoencoder_model.hdf5'
checkpoint_ssemnet_name = '_ssemnet_model.hdf5'

checkpoint_autoencoder_filename = path.join(
    checkpoint_dir, dataset_name + checkpoint_autoencoder_name)
checkpoint_ssemnet_filename = path.join(
    checkpoint_dir, dataset_name + checkpoint_ssemnet_name)

image_sink = 'results/'

encoding_decoding_choice = 1
# encoding_decoding_choice = None
