"""Estimate stroke features."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pdb


def get_line(X, Y, pt, flag):
    """Generate equation for a line given a set of points."""
    if X[1]-X[0] == 0:
        return X[0]
    m = ((Y[1]-Y[0])*pt)/float(X[1]-X[0])
    c = Y[0] - m*X[0]
    if flag == "x":
        return m*pt + c
    else:
        return float(pt - c)/m


def feature(img, n):
    """."""
    y, x = (img == 0).nonzero()
    feature = []
    for i in np.linspace(0, img.shape[1]-1.1, n):
        lb = np.floor(i)
        ub = lb + 1
        try:
            lb_index = (x == lb).nonzero()[0][0]
            ub_index = (x == ub).nonzero()[0][0]
        except:
            pdb.set_trace()
        j = get_line([lb, ub], [y[lb_index], y[ub_index]], i, "x")
        feature.append(j)

    for j in np.linspace(0, img.shape[0]-1.1, n):
        lb = np.floor(j)
        ub = lb + 1
        lb_index = (y == lb).nonzero()[0][0]
        ub_index = (y == ub).nonzero()[0][0]
        i = get_line([x[lb_index], x[ub_index]], [lb, ub], j, "y")
        feature.append(i)
    return feature


data_path = "/home/sanny/honours/Texture_Analysis/output/"
folderlist = os.listdir(data_path)
folderlist.sort()

output_file = open('features_xy.csv', 'w')
for name in folderlist:
    if name[-5:] == '.tiff':
        print("Processing " + name)
        img = cv2.imread(data_path+name, 0)[1:-1, 1:-1]
        feat = feature(img, 25)
        f = ','.join(str(val) for val in feat)
        output_file.write(str(f)+","+name+'\n')
output_file.close()
