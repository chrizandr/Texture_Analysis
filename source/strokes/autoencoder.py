"""Autoencoder for stroke features."""

import numpy as np
from keras.preprocessing import image
from keras.layers import Dense, Input
from keras.models import Model
from random import shuffle
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import load_model
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
            self.indexes = self.__get_exploration_order()
            imax = int(len(self.list_IDs)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.list_IDs[k] for k in self.indexes[i*self.batch_size:(i+1)*self.batch_size]]
                X = self.__data_generation(list_IDs_temp)
                yield (X, X)
            list_IDs_temp = [self.list_IDs[k] for k in self.indexes[imax*self.batch_size:len(self.list_IDs)]]
            X = self.__data_generation(list_IDs_temp)
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
            try:
                x = np.concatenate((x, 255*np.ones(shape)), axis)
            except:
                pdb.set_trace()
            x = x/255
            x = 1 - x
            X[i, :] = x.reshape(self.dim*self.dim)
        return X


def partition_data(data_path):
    """Get the files from the data path."""
    files = {
        "train": [],
        "test": [],
        "validation": []
    }
    folders = os.listdir(data_path)
    for folder in folders:
        print(data_path + folder + '/')
        f = os.listdir(data_path + folder + '/')
        f = [folder+'/'+x for x in f]
        shuffle(f)
        test_partition = int(len(f)*0.8)
        train = f[0:test_partition]
        test = f[test_partition::]
        files["train"].extend(train)
        files["test"].extend(test)
    shuffle(files["train"])
    shuffle(files["test"])
    validation_partition = len(files["train"]) - 1000
    files["validation"] = files["train"][validation_partition::]
    files["train"] = files["train"][0:validation_partition]
    return files


def autoencoder_model(encoding_dim, input_shape):
    """Create autoencoder model."""
    input_img = Input(shape=(input_shape[0]*input_shape[1],))
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    decoded = Dense(input_shape[0]*input_shape[1], activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    return autoencoder


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


if __name__ == '__main__':
    filepath = "/home/chrizandr/data/Telugu/strokes/"
    output_file = "autoencoder_.hd5"
    batch_size = 32
    encoding_dim = 100
    input_dim = (25, 25)

    print("Partitioning data")
    data = partition_data(filepath)

    training_generator = DataGenerator(dim=input_dim[0], batch_size=batch_size, shuffle=True,
                                       list_IDs=data["train"], filepath=filepath).generator()
    print("Reading validation data")
    validation_data = get_data(len(data["validation"]), input_dim[0], data["validation"], filepath)

    autoencoder = autoencoder_model(encoding_dim, input_dim)
    # autoencoder = load_model(output_file)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath=output_file, monitor='val_loss')
    tboard = TensorBoard(log_dir="logs/")

    autoencoder.fit_generator(generator=training_generator,
                              steps_per_epoch=len(data['train'])//batch_size,
                              validation_data=validation_data,
                              validation_steps=len(data['validation'])//batch_size,
                              callbacks=[checkpoint, tboard],
                              epochs=20)
    test_generator = DataGenerator(dim=input_dim[0], batch_size=batch_size, shuffle=True,
                                   list_IDs=data["test"], filepath=filepath).generator()
    model = load_model(output_file)
    results = model.evaluate_generator(generator=test_generator, steps=len(data["test"])//batch_size)
    print(results)
