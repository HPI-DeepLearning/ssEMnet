import os

image_1_dir = 'data/mnist_min/0_1'
image_2_dir = 'data/mnist_min/0_2'
processed_dir = 'data/processed'
checkpoint_dir = 'checkpoints'
checkpoint_name = 'mynet'
checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_name)
encoding_decoding_choice = 1
image_sink = 'results/'
concatenated_filename = os.path.join(processed_dir, 'concatenated')
