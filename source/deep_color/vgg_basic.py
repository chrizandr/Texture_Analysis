# -*- coding: utf-8 -*-
"""VGG16 model for Keras."""
from __future__ import print_function

import numpy as np
import os
import csv
import pdb

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Flatten
from keras.layers import Dense


class DataGenerator(object):
    """Generates data for Keras."""

    def __init__(self, dim_x=224, dim_y=224, dim_z=3, batch_size=32, shuffle=True, n_classes=150, list_IDs=[], labels=[], filepath=""):
        """Initialization."""
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.list_IDs = list_IDs
        self.labels = labels

    def generate(self, labels):
        """Generate batches of samples."""
        while 1:
            indexes = self.get_exploration_order(self.list_IDs)

            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [self.list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                X, y = self.data_generation(labels, list_IDs_temp)

                yield X, y

    def get_exploration_order(self):
        """Generate order of exploration."""
        indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def data_generation(self, labels, list_IDs_temp):
        """Generate data of batch_size samples."""
        X = np.empty((self.batch_size, 1, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            img = image.load_img(filepath + ID, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X[i, 0, :, :, :] = x
            name = ID.split('-')[0] + '-' + ID.split('-')[1]
            y[i] = labels[name]

        return X, self.sparsify(y)

    def sparsify(self, y):
        """Return labels in binary NumPy array."""
        # y[i] == j+1 if labels start from 0, else y[i] == j
        return np.array([[1 if y[i] == j+1 else 0 for j in range(self.n_classes)]
                         for i in range(y.shape[0])])


def getIds(filename):
    """Return dictionary of writers."""
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ids[row[0][0:-4]] = int(row[1])
    return ids


def VGG_Writer(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):
    """VVG16 without final dense layers."""
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)
    # pdb.set_trace()
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    feature_part = Model(inputs, x, name='vgg16')

    weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    feature_part.load_weights(weights_path)

    img_input = Input(shape=(224, 224, 3))
    features = feature_part(img_input)
    x = Flatten(name='flatten')(features)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(150, activation='softmax', name='predictions')(x)

    model = Model(img_input, predictions, name="classifier")

    return model


def create_Partition(filepath, labels):
    """Create training and validation sets from files."""
    data = dict()
    x = os.listdir(filepath)
    data["training"] = x

    validation = list()
    for l in labels:
        for f in x:
            name = f.split('-')[0] + '-' + f.split('-')[1]
            if name == l:
                validation.append(f)
                break

    data["validation"] = validation

    return data


if __name__ == '__main__':
    filepath = "/home/chrizandr/data/Telugu/test/"
    class_path = "/home/chrizandr/data/writerids.csv"
    output_file = "/home/chrizandr/data/VGG_Writer.hd5"
    batch_size = 32

    labels = getIds(class_path)
    data = create_Partition(filepath, labels)

    checkpoint = ModelCheckpoint(filepath=output_file, monitor='val_loss')

    model = VGG_Writer(include_top=False, weights='imagenet')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_generator = DataGenerator(dim_x=224, dim_y=224, dim_z=3, batch_size=batch_size, shuffle=True,
                                       n_classes=150, list_IDs=data["training"], labels=labels, filepath=filepath)

    validation_generator = DataGenerator(dim_x=224, dim_y=224, dim_z=3, batch_size=batch_size, shuffle=True,
                                         n_classes=150, list_IDs=data["validation"], labels=labels, filepath=filepath)

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(data['train'])//batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(data['validation'])//batch_size)
