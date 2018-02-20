"""Calcullate the center of mass based features for stroke representations."""
import numpy as np
from scipy import spatial
import math
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import os
import cv2
import pdb

'''
if  n= 6
sm = [nan nan] check that
'''


def strip_white(img):
    """Remove white parts surrounding the image."""
    m, n = img.shape
    proj = np.sum(img, axis=1)
    cords = (proj != (255*n)).nonzero()
    proj_2 = np.sum(img, axis=0)
    cords_2 = (proj_2 != (255*m)).nonzero()
    img = img[cords[0][0]:cords[0][-1]+1, cords_2[0][0]:cords_2[0][-1]+1]
    return img


def get_feature(img, l=6):
    """Return the feature calculated for an image."""
    # plt.imshow(img)
    # plt.show()
    img = strip_white(img)
    m, n = img.shape
    img = 255 - img
    cm = np.array(center_of_mass(img)) + 1
    pixel_list = np.linspace(0, n, num=l).astype(int)

    sub_imgs = []
    for i in range(l-1):
        sub_img = np.zeros(img.shape)
        sub_img[:, pixel_list[i]: pixel_list[i+1]] = img[:, pixel_list[i]: pixel_list[i+1]]
        sub_imgs.append(sub_img)

    feature = []
    cms = []
    for i in range(len(sub_imgs)):
        each = sub_imgs[i]
        sm = np.array(center_of_mass(each)) + 1
        dist = np.linalg.norm(cm-sm)
        cosine_ = spatial.distance.cosine(cm, sm)
        feature.append(dist)
        try:
            feature.append(cosine_)
        except ValueError:
            pdb.set_trace()
        if math.isnan(dist) or math.isnan(cosine_):
            pdb.set_trace()
        cms.append(sm)

    # img3 = np.dstack([img, img, img])
    # img3[int(cm[0]), int(cm[1]), :] = (0, 255, 0)
    # for each in cms:
    #    i = int(each[0])
    #    j = int(each[1])
    #    img3[i, j, :] = (255, 0, 0)
    # plt.imshow(img3)
    # plt.show()

    return feature


if __name__ == "__main__":

    path = "/home/chrizandr/data/Telugu/strokes/"
    output_file = open("cg_features.csv", "w")
    sub_dirs = os.listdir(path)

    for dr in sub_dirs:
        print("Processing", dr)
        dr_path = path + dr
        files = os.listdir(dr_path)
        for each in files:
            file_name = dr_path + "/" + each
            img = cv2.imread(file_name, 0)
            fea = get_feature(img)
            feat_str = ','.join(str(val) for val in fea)
            output_file.write(feat_str + ',' + dr + '-' + each + '\n')

    output_file.close()
