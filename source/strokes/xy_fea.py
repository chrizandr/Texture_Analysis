"""Estimate stroke features."""

import numpy as np
import cv2
import os
import pdb
import math
from visualisor import printProgressBar


def get_line(X, Y, pt, flag):
    """Generate equation for a line given a set of points."""
    if X[1]-X[0] == 0:
        return X[0]
    m = (Y[1]-Y[0])/float(X[1]-X[0])
    c = Y[0] - m*X[0]
    output = 0
    if flag == "x":
        output = m*pt + c
    else:
        output = float(pt - c)/m
    if math.isnan(output):
        pdb.set_trace()
    return output


def feature(img, n, m):
    """."""
    y, x = (img == 0).nonzero()
    feature = []
    for i in np.linspace(0, img.shape[1]-2, n):
        lb = np.floor(i)
        ub = lb + 1
        try:
            lb_index = (x == lb).nonzero()[0][0]
            ub_index = (x == ub).nonzero()[0][0]
        except:
            print(x, y, lb, ub)
            print("Exception in feature()!")
            pdb.set_trace()

        j = get_line([lb, ub], [y[lb_index], y[ub_index]], i, "x")
        feature.append(j)

    for j in np.linspace(0, img.shape[0]-2, m):
        lb = np.floor(j)
        ub = lb + 1
        lb_index = (y == lb).nonzero()[0][0]
        ub_index = (y == ub).nonzero()[0][0]
        i = get_line([x[lb_index], x[ub_index]], [lb, ub], j, "y")
        feature.append(i)
    return feature


if __name__ == "__main__":
    data_path = "/home/chrizandr/data/Telugu/strokes/"
    folders = os.listdir(data_path)
    folders.sort()

    # img = cv2.imread(data_path + folder + '/' + stroke, 0)[1:-1, 1:-1]
    i = 0
    f = open("linear_features.csv", "w")
    for folder in folders:
        printProgressBar(i+1, 450, "Progress", "Complete", length=50)
        i += 1
        strokes = os.listdir(data_path + folder + '/')
        # print("Processing " + folder)
        for stroke in strokes:
            if stroke[-5:] == '.tiff':
                img = cv2.imread(data_path + folder + '/' + stroke, 0)[1:-1, 1:-1]
                feat = feature(img, 25, 25)
                feat_str = ','.join(str(val) for val in feat)
                f.write(feat_str + ',' + folder + '-' + stroke + '\n')
    f.close()
