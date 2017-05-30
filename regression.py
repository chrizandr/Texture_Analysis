from sklearn import svm
import cv2
import numpy as np
import pdb
from scipy import ndimage
import matplotlib.pyplot as plt
import os

def get_line(X, Y):
    A = np.vstack([X, np.ones(len(X))]).T
    return np.linalg.lstsq(A, Y)[0]

def feature(img):
    x,y = (img==0).nonzero()
    X = [[each] for each in x]
    clf = svm.SVR(kernel='poly')
    clf.fit(X, y)

    #A = [0, img.shape[0]]
    #B1 = [0, img.shape[1]]
    #B2 = [img.shape[1], 0]
    #print img.shape
    #m1, c1 = get_line(A, B1)
    #m2, c2 = get_line(A, B2)
    #print m1,c1,m2,c2
    point_y = []
    point_x = []
    for i in range(0,img.shape[0],img.shape[0]/10):
        point_x.append(i)
        point_y.append(clf.predict(i))

    plt.scatter(x,y)
    plt.scatter(point_x, point_y, color = 'red')
    plt.show()


data_path = "/home/sanny/Documents/clustering_model/data/"
folderlist = os.listdir(data_path)
folderlist.sort()

for name in folderlist:
    if name[-4:]=='.png':
        print("Processing "+ name)
        img = cv2.imread(data_path+name, 0)
        feature(img)
