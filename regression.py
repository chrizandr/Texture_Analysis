"""Regressor for estimating stroke curves."""

from sklearn import svm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb


def get_line(X, Y):
    """Generate equation for a line given a set of points."""
    A = np.vstack([X, np.ones(len(X))]).T
    return np.linalg.lstsq(A, Y)[0]


def feature(img):
    """Genereate curve features."""
    print("Getting points...")
    y, x = (img == 0).nonzero()
    print("Training...")
    X = x.reshape(-1, 1)
    clf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    clf.fit(X, y)
    print("Predicting...")
    point_y = []
    point_x = []
    for i in np.linspace(0, img.shape[1], 100):
        point_x.append(i)
        point_y.append(int(clf.predict(i)))
    #plt.scatter(x, y)
    #plt.scatter(point_x, point_y, color='red')
    #plt.show()
    return point_y

data_path = "/home/sanny/honours/Texture_Analysis/output/"
folderlist = os.listdir(data_path)
folderlist.sort()

output_file = open('features.csv', 'w')

for name in folderlist:
    if name[-5:]=='.tiff':
        print("Processing "+ name)
        img = cv2.imread(data_path+name, 0)
        f = ','.join(str(val) for val in feature(img))
        output_file.write(str(f)+","+name+'\n')
output_file.close()
