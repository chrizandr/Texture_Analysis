"""File for clustering strokes."""

import numpy as np
import csv
import os
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn import metrics
# from collections import Counter
import pickle
from shutil import copyfile
import pdb


def cluster(X, n_clusters, file_names, out_folder="", src_folder="", output=False, score=False):
    """Cluster strokes, calculate distribution, group images according to clusters."""
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=20000, batch_size=5000, init_size=10000)

    print("Clustering the strokes")
    kmeans.fit(X)
    output_labels = kmeans.labels_

    if score:
        samples = np.random.choice(X.shape[0], 10000, replace=False)
        X_test = X[samples]
        y_test = output_labels[samples]
        score = metrics.silhouette_score(X_test, y_test, metric='euclidean')
        return score

    if output:
        for i in range(0, n_clusters):
            print("Processing cluster {}".format(i))
            os.mkdir(out_folder + str(i))
            file_indices = (output_labels == i).nonzero()[0]
            for j in file_indices:
                name = file_names[j]
                parts = name.split('-')
                src = src_folder + "-".join(parts[0:2]) + '/' + parts[2]
                dest = out_folder + str(i) + '/' + name
                copyfile(src, dest)

    return output_labels


def read_data(feature_file, normalize=False, shuffle=True):
    """Read and normalise data from a feature file."""
    X = list()
    strokes = list()
    reader = csv.reader(open(feature_file, "r"))

    for row in reader:
        row_ = [100*float(x) for x in row[0:-1]]
        if normalize:
            max_val = max(row_)
            X.append([x/max_val for x in row_])
        else:
            X.append(row_)
        strokes.append(row[-1])

    X = np.array(X)
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        new_strokes = [strokes[i] for i in indices]
        return X, new_strokes

    return X, strokes


def find_scores(FEATURE_FILE, range_=(20, 100)):
    """Find the clustering score for different k."""
    # Best scores 32, 37, 39, 40, 49, 63

    print("Reading data...")
    X, strokes = read_data(FEATURE_FILE)

    scores = []
    for i in range(range_[0], range_[1]):
        print("Clustering k = {}".format(i))
        scores.append(cluster(X, i, strokes, score=True))

    return scores


if __name__ == "__main__":
    FEATURE_FILE = "output.csv"
    OUT_FOLDER = "/home/chrizandr/data/Telugu/clustered_strokes_31/"
    SRC_FOLDER = "/home/chrizandr/data/Telugu/strokes/"
    RANGE = (20, 100)
    # scores = find_scores(FEATURE_FILE, RANGE)

    print("Reading data...")
    X, strokes = read_data(FEATURE_FILE)
    cluster(X, 31, strokes, OUT_FOLDER, SRC_FOLDER, output=True)

    pdb.set_trace()
