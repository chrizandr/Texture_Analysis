"""Find representations for clusters."""

import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import pdb


def get_data(size, dim, IDs, filepath):
    """Read data."""
    X = np.empty((size, dim*dim))

    for i, ID in enumerate(IDs):
        img = image.load_img(filepath + ID, grayscale=True)
        size = img.size
        if size[0] > size[1]:
            height = dim
            width = int(dim * size[1] / size[0])
            shape = (dim - width, dim)
            axis = 0
        else:
            width = dim
            height = int(dim * size[0] / size[1])
            shape = (dim, dim - height)
            axis = 1
        if height <= 0 or width <= 0:
            continue
        img = img.resize((height, width))
        x = image.img_to_array(img)[:, :, 0]
        try:
            x = np.concatenate((x, 255*np.ones(shape)), axis)
        except:
            pdb.set_trace()
        x = x/255
        x = 1 - x
        X[i, :] = x.reshape(dim*dim)
    return (X, X)


def partition_data(data_path):
    """Get the files from the data path."""
    f = os.listdir(data_path)
    files = [data_path + x for x in f]
    return files


ENCODER = "autoencoder_tel.hd5"
CLUSTER = ""
images = partition_data("rep/")
X, y = get_data(len(images), 25, images, "")
autoencoder = load_model(ENCODER)

# input_layer = autoencoder.input
# output_layer = autoencoder.layers[1].output
# encoder = Model(input_layer, output_layer)

output = autoencoder.predict(X)
# pdb.set_trace()
# plt.subplot(121)
# plt.plot(output[0])
# plt.subplot(122)
# plt.plot(output[1])
# plt.show()
#
for i in range(X.shape[0]):
    plt.subplot(241 + i)
    plt.imshow(1-X[i].reshape(25, 25), 'gray')

for i in range(X.shape[0]):
    plt.subplot(245 + i)
    plt.imshow(1-output[i].reshape(25, 25), 'gray')
plt.show()
