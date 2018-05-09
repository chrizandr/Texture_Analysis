import numpy as np
from skimage import feature as skfeature
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy import signal, stats
import cv2
import os
import time
import pdb


def getangle(n):
    """Angle in radians."""


def angular_moment(x):
    """Angular moment."""
    return np.sum(x*x)


def contrast(x):
    """Contrast."""
    m, n = x.shape
    y = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            y[i, j] = (i-j)*(i-j)
    return np.sum(y*x)


def entropy(x):
    """Entropy."""
    nonzeros = x.nonzero()
    new_vals = x[nonzeros[0], nonzeros[1]]
    return -np.sum(new_vals*np.log(new_vals))


def correlation(x):
    """Correlation."""
    nonzeros = x.nonzero()
    new_vals = x[nonzeros[0], nonzeros[1]]
    return np.sum(np.corrcoef(new_vals, new_vals))


def feature_set(name, distances, angles):
    """Feature."""
    img = cv2.imread(name, 0)
    kernel = np.ones((3, 3), np.uint8)
    dilation = 255-cv2.dilate(255-img, kernel, iterations=1)

    features = list()
    x = skfeature.greycomatrix(dilation, distances, angles, levels=256, symmetric=True)
    for d in range(len(distances)):
        for theta in range(len(angles)):
            f = [angular_moment(x[:, :, d, theta]), contrast(x[:, :, d, theta]),
                 entropy(x[:, :, d, theta]), correlation(x[:, :, d, theta])]
            if np.isnan(f).any():
                pdb.set_trace()
            features.extend(f)
    return features


if __name__ == "__main__":
    # ------------------------------ Global variables for GSCM features.
    distances = [1, 2, 3, 4, 5]         # The distances of the various GSCM matrices to be computed
    angles = [0, 45, 90, 135]          # The orientations of the various GSCM matrices to be computed
    angles = [getangle(x) for x in angles]
    # ------------------------------

    lang = "firemaker"
    data_path = ["/home/chrizandr/data/"+lang+"/linear_color_maps/", "/home/chrizandr/data/"+lang+"/auto_enc_color_maps/", "/home/chrizandr/data/"+lang+"/cmass_color_maps/"]
    output_file = ["/home/chrizandr/data/"+lang+"/linear_color_maps_GSCM.csv", "/home/chrizandr/data/"+lang+"/auto_enc_color_maps_GSCM.csv", "/home/chrizandr/data/"+lang+"/cmass_color_maps_GSCM.csv"]

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
                A = feature_set(img_name, distances, angles)
                # ------------------------------
                for feature in A:
                    f.write(str(feature)+',')
                f.write(name[0:-4] + '\n')
                print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
        f.close()
