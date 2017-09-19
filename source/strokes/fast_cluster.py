"""."""
import numpy as np
from sklearn.cluster import KMeans
import pdb
import matplotlib.pyplot as plt
import cv2
from shutil import copyfile
import os
import operator


def cluster(X):
    """."""
    print X.shape
    kmeans = KMeans(n_clusters=97,
                    init='k-means++')
    for i in range(0, X.shape[0], 2000):
        print X[i:i+2000]
        kmeans.fit(X[i:i+2000])
        output = kmeans.labels_

    return 1


print("Getting from file...")
f = open('features_xy.csv', 'r')
file_name = []
fea = []
for line in f:
    line = line.strip().split(',')
    file_name.append(line[-1])
    fea.append([float(each) for each in line[:-1]])

print("Converting to array")
X = np.array(fea)
lis = []
v = cluster(X)
