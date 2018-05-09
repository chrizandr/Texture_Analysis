import numpy as np
from skimage import feature as skfeature
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy import signal, stats
import cv2
import os
import time
import pdb
from skimage.feature import local_binary_pattern,multiblock_lbp
import cv2
import numpy as np


def lbp_feature_set(img):
    """LBP Feature."""
    limg = local_binary_pattern(img, 8, 2, 'default')
    limg = limg.astype(np.uint8)
    m, n = limg.shape
    feature = [0 for i in range(0, 256)]
    for i in range(0, m):
        for j in range(0, n):
            feature[img[i, j]] += 1
    feat = np.histogram(limg.ravel(), 256, [0, 256])[0]
    return feat[0:-1]


def feature_set(name):
    """Feature."""
    img = cv2.imread(name, 0)
    kernel = np.ones((3, 3), np.uint8)
    dilation = 255-cv2.dilate(255-img, kernel, iterations=1)

    features = lbp_feature_set(dilation)
    return features


if __name__ == "__main__":
    lang = "Kannada"
    data_path = ["/home/chrizandr/data/"+lang+"/linear_color_maps/", "/home/chrizandr/data/"+lang+"/auto_enc_color_maps/", "/home/chrizandr/data/"+lang+"/cmass_color_maps/"]
    output_file = ["/home/chrizandr/data/"+lang+"/linear_color_maps_lbp.csv", "/home/chrizandr/data/"+lang+"/auto_enc_color_maps_lbp.csv", "/home/chrizandr/data/"+lang+"/cmass_color_maps_lbp.csv"]

    for i, path in enumerate(data_path):
        f = open(output_file[i], "w")
        # Loop over the files in the dataset

        folderlist = os.listdir(path)
        folderlist.sort()
        for name in folderlist:
            if name[-4:] in ['.png', '.jpg', '.tif']:
                print("Processing " + name)

                start_time = time.time()          # Code for time measurement
                img_name = path + name
                # ------------------------------ The feature set function from the appropriately imported feature source is used.
                A = feature_set(img_name)
                # ------------------------------
                feat_str = ",".join([str(x) for x in A])
                f.write(feat_str + ',')
                f.write(name[0:-4] + '\n')
                print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
        f.close()
