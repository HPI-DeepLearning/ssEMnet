import os

image_1_dir = 'data/Cardiac_small_sample/ES_moving' #moving
image_2_dir = 'data/Cardiac_small_sample/ED_fixed' #fixed
processed_dir = 'data/processed'
checkpoint_dir = 'checkpoints'
checkpoint_name = 'autoencoderModel'
checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_name)
encoding_decoding_choice = 1
image_sink = 'results/'
concatenated_filename = os.path.join(processed_dir, 'concatenated')
ModelFile = r'models/ssemnet.hdf5'
