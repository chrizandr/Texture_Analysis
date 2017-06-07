"""File for clustering strokes."""

import numpy as np
import csv
from sklearn.cluster import KMeans
import pdb
from shutil import copyfile
import os


def cluster(X, n_clusters, file_name):
    """Cluster strokes, calculate distribution, group images according to clusters."""
    print("Clustering the strokes...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', ).fit(X)
    output = kmeans.labels_
    folders = set(output)
    print("Making stroke folders...")
    for f in folders:
        os.makedirs("/home/chris/data/clusters/" + str(f) + '/')
    print("Calculating distributions and copying files...")
    for i in range(X.shape[0]):
        key = file_name[i].split('-')[0] + '-' + file_name[i].split('-')[1]
        keymap[key].append(output[i])
        src = "/home/chris/data/strokes/" + key + '/' + file_name[i].split('-')[2]
        dest = "/home/chris/data/clusters/" + str(output[i]) + '/' + file_name[i]
        copyfile(src, dest)

    return None


def read_data(feature_file):
    """Read and normalise data from a feature file."""
    X = list()
    strokes = list()
    reader = csv.reader(open(feature_file, "rb"))
    for row in reader:
        row_ = [float(x) for x in row[0:-1]]
        max_val = max(row_)
        X.append([x/max_val for x in row_])
        strokes.append(row[-1])
    X = np.array(X)
    return X, strokes


feature_file = "features.csv"
print("Reading data...")
X, strokes = read_data(feature_file)
files = set([x.split('-')[0] + '-' + x.split('-')[1] for x in strokes])
keymap = dict()
for f in files:
    keymap[f] = list()

cluster(X, 100, strokes)
pdb.set_trace()
