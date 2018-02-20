"""Encoder for the strokes."""

from keras.models import load_model
from keras.models import Model
import numpy as np
from keras.preprocessing import image
import os
import pdb


class DataGenerator(object):
    """Generates data for Keras."""

    def __init__(self, dim=25, batch_size=32, shuffle=False, list_IDs=[], filepath=""):
        """Initialization."""
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.filepath = filepath
        self.indexes = self.__get_exploration_order()
        self.current = 0

    def generator(self):
        """Generate batches of samples."""
        while 1:
            imax = int(len(self.list_IDs)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.list_IDs[k] for k in self.indexes[i*self.batch_size:(i+1)*self.batch_size]]
                X = self.__data_generation(list_IDs_temp)
                yield (X, X)
            list_IDs_temp = [self.list_IDs[k] for k in self.indexes[imax*self.batch_size:len(self.list_IDs)]]
            X = self.__data_generation(list_IDs_temp)
            self.indexes = self.__get_exploration_order()
            yield (X, X)

    def __get_exploration_order(self):
        """Generate order of exploration."""
        indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, list_IDs_temp):
        """Generate data of batch_size samples."""
        X = np.zeros((self.batch_size, self.dim * self.dim))

        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(self.filepath + ID, grayscale=True)
            size = img.size
            if size[0] > size[1]:
                height = self.dim
                width = int(self.dim * size[1] / size[0])
                shape = (self.dim - width, self.dim)
                axis = 0
            else:
                width = self.dim
                height = int(self.dim * size[0] / size[1])
                shape = (self.dim, self.dim - height)
                axis = 1
            if height <= 0 or width <= 0:
                continue
            img = img.resize((height, width))
            x = image.img_to_array(img)[:, :, 0]
            x = np.concatenate((x, 255*np.ones(shape)), axis)
            x = x/255
            x = 1 - x
            X[i, :] = x.reshape(self.dim*self.dim)
        return X


def get_data(data_path):
    """Get the files from the data path."""
    data = []
    folders = os.listdir(data_path)
    for folder in folders:
        print(data_path + folder + '/')
        f = os.listdir(data_path + folder + '/')
        f = [folder+'/'+x for x in f]
        data.extend(f)
    return data


def read_data(size, shape, IDs, filepath):
    """Read data."""
    X = np.empty((size, shape[0]*shape[1]))

    for i, ID in enumerate(IDs):
        img = image.load_img(filepath + ID, target_size=(shape[0], shape[1]), grayscale=True)
        x = image.img_to_array(img)[:, :, 0]
        x = x/255
        x = 1 - x
        X[i, :] = x.reshape(shape[0]*shape[1])

    return (X, X)


if __name__ == "__main__":
    model_file = "autoencoder.hd5"
    filepath = "/home/chrizandr/data/Telugu/strokes/"
    batch_size = 32

    model = load_model(model_file)

    input_layer = model.input
    output_layer = model.layers[1].output
    encoder = Model(input_layer, output_layer)

    data = get_data(filepath)
    data_gen = DataGenerator(dim=25, batch_size=batch_size, shuffle=False,
                             list_IDs=data, filepath=filepath).generator()
    print("Predicting")
    predictions = encoder.predict_generator(generator=data_gen, steps=len(data)//batch_size + 1)

    print("Writing the output")
    f = open("auto_enc_feature.csv", "w")
    for i in range(len(data)):
        f.write(",".join([str(x) for x in predictions[i]]))
        f.write("," + data[i].split('/')[0] + '-' + data[i].split('/')[1] + "\n")
    f.close()
