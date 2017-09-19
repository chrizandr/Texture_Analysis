"""File for clustering strokes."""

import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans as KMeans
from collections import Counter
import pickle


def cluster(X, n_clusters, file_name, keymap=dict(), distributions=False):
    """Cluster strokes, calculate distribution, group images according to clusters."""
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=20000, batch_size=5000, init_size=10000)

    print("Clustering the strokes")
    kmeans.fit(X)

    print("Saving the model")
    pickle.dump(kmeans, open('cluster.pkl', 'wb'))

    if distributions:
        f = open("distributions.csv", "w")

        for key in keymap:
            count = Counter(keymap[key])
            feat = list()
            for i in range(n_clusters):
                if i in count:
                    feat.append(count[i])
                else:
                    feat.append(0)
            feat_str = ','.join(str(val) for val in feat)
            f.write(feat_str + ',' + key + '\n')

    return None


def read_data(feature_file):
    """Read and normalise data from a feature file."""
    X = list()
    strokes = list()
    reader = csv.reader(open(feature_file, "r"))

    for row in reader:
        row_ = [float(x) for x in row[0:-1]]
        max_val = max(row_)
        X.append([x/max_val for x in row_])
        strokes.append(row[-1])

    X = np.array(X)
    indices = np.random.permutation(np.shape[0])
    X = X[indices]
    strokes = strokes[indices]
    return X, strokes


if __name__ == "__main__":
    FEATURE_FILE = "features.csv"

    print("Reading data...")
    X, strokes = read_data(FEATURE_FILE)

    files = set([x.split('-')[0] + '-' + x.split('-')[1] for x in strokes])
    keymap = dict()

    for f in files:
        keymap[f] = list()

    cluster(X, 58, strokes)
