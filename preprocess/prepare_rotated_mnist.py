from os import path, walk, makedirs
from random import randint
from skimage import io, transform

def get_filenames(index_start, index_end):
    for root, dirs, files in walk('data/raw/mnist_png/training/1'):
      return [path.join('data/raw/mnist_png/training/1', fn) for fn in files[index_start:index_end]]


for index, fn in enumerate(get_filenames(0, 100)):
    image = io.imread(fn)
    # image = transform.rotate(image, randint(0, 90))

    dir = 'data/mnist/training/moving/'
    if not path.exists(dir):
        makedirs(dir)
    
    print(image.shape)
    print(image)
    io.imsave(dir + str(index) + '.png', image)


for index, fn in enumerate(get_filenames(100, 200)):
    image = io.imread(fn)
    # image = transform.rotate(image, randint(0, 360))

    dir = 'data/mnist/training/fixed/'
    if not path.exists(dir):
        makedirs(dir)
    io.imsave(dir + str(index) + '.png', image)
