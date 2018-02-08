"""Data Generator."""
import os
import pdb
from keras.preprocessing import image
import numpy as np
import csv


class DataGenerator(object):
    """Generates data for Keras."""

    def __init__(self, slen=41, dim_x=224, dim_y=224, dim_z=3, batch_size=1, shuffle=True, n_classes=150):
        """Init."""
        self.slen = slen
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes

    def generate(self, list_IDs, labels):
        """Generate batches of samples."""
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                labels_temp = [labels[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                X, y = self.__data_generation(labels_temp, list_IDs_temp)
                yield (X, y)

            list_IDs_temp = [list_IDs[k] for k in indexes[imax*self.batch_size:len(list_IDs)]]
            labels_temp = [labels[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
            X, y = self.__data_generation(labels_temp, list_IDs_temp)
            yield (X, y)

    def __get_exploration_order(self, list_IDs):
        """Generate order of exploration."""
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        """Generate data of batch_size samples."""
        # X : (n_samples, slen, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.slen, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            for j, id_ in enumerate(ID):
                img = image.load_img(id_)
                img = image.img_to_array(img)
                X[i, j, :, :, :] = img

            y[i] = labels[i]

        return X, sparsify(y, self.n_classes)


def sparsify(y, n_classes):
    """Return labels in binary NumPy array."""
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                    for i in range(y.shape[0])])


def getIds(filename):
    """Get ids of file."""
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ids[row[0]] = int(row[1])
    return ids


def partition_data(path, writer_ids):
    """Read the data."""
    partition = {}
    ids = {}
    temp_data = []
    temp_ids = []

    sessions = os.listdir(path)
    for s in sessions:
        sess_path = path + s + '/'
        pages = os.listdir(sess_path)
        for p in pages:
            page_path = sess_path + p + '/'
            sentences = os.listdir(page_path)
            for sent in sentences:
                sent_images = []
                sentences_path = page_path + sent + '/'
                parts = os.listdir(sentences_path)
                for part in parts:
                    name = sentences_path + part
                    id_ = sent.split('-')[0] + '-' + sent.split('-')[1]
                    sent_images.append(name)
                temp_data.append(sent_images)
                temp_ids.append(writer_ids[id_])

    indices = np.arange(len(temp_data))
    np.random.shuffle(indices)
    temp_data = [temp_data[k] for k in indices]
    temp_ids = [temp_ids[k] for k in indices]

    train_part = int(len(temp_data) * 0.8)
    val_part = int(train_part * 0.8)

    partition["train"] = temp_data[0:val_part]
    partition["validation"] = temp_data[val_part:train_part]
    partition["test"] = temp_data[train_part::]

    ids["train"] = temp_ids[0:val_part]
    ids["validation"] = temp_ids[val_part:train_part]
    ids["test"] = temp_ids[train_part::]

    return partition, ids


if __name__ == "__main__":
    PATH = "/home/chrizandr/data/new_data/"
    ID_FILE = "/home/chrizandr/data/IAM/IAM_writerids.csv"

    writer_ids = getIds(ID_FILE)
    partition, ids = partition_data(PATH, writer_ids)
    training_generator = DataGenerator(slen=41, dim_x=224, dim_y=224,
                                       dim_z=3, batch_size=1,
                                       shuffle=True, n_classes=657).generate(partition['train'], ids["train"])
    X, y = next(training_generator)
    print(y.nonzero()[1])
    pdb.set_trace()
