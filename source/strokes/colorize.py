"""Extract strokes from a given image."""

import cv2
from skimage.morphology import skeletonize_3d
# import matplotlib.pyplot as plt
import pickle
# import pdb
import numpy as np
import os
import multiprocessing
import random
from xy_fea import feature


def get_color_map(n):
    """Return color map for given number of colors."""
    random.seed(10)
    colormap_img = np.ones((5*n, 5*n, 3)) * 255

    colors = np.random.permutation(3*n).reshape(3, n).astype(np.float32)
    color_map = (255 * colors/(n*3)).astype(np.int32)

    k = 0
    for i in range(n):
        colormap_img[k:k+5, :, ] = color_map[:, i]
        k += 5

    return np.array(color_map)


def colorize_strokes(activated, cluster, color_map):
    """Colorize strokes based on clustering."""
    nimg = cv2.connectedComponents(1-activated)[1]
    colored_img = np.ones((activated.shape[0], activated.shape[1], 3)) * 255
    # Get the connectedComponents
    labels = set(np.unique(nimg))
    labels.remove(0)
    for label in labels:
        sub_region = (nimg == label).nonzero()
        max_hor = sub_region[1].max()
        min_hor = sub_region[1].min()
        max_ver = sub_region[0].max()
        min_ver = sub_region[0].min()
        img = np.ones((max_ver-min_ver+1, max_hor-min_hor+1), dtype=np.uint8)
        img[sub_region[0]-min_ver, sub_region[1]-min_hor] = 0
        if max_hor - min_hor > 5 and max_ver - min_ver > 5:
            img = np.vstack((img, np.ones((1, img.shape[1]))))
            img = np.vstack((np.ones((1, img.shape[1])), img))
            img = np.hstack((img, np.ones((img.shape[0], 1))))
            img = np.hstack((np.ones((img.shape[0], 1)), img))
            img = (img*255)[1:-1, 1:-1]

            raw_feature = feature(img, 25, 25)
            max_val = max(raw_feature)
            norm_feature = [x/max_val for x in raw_feature]

            prediction_label = cluster.predict(np.array(norm_feature).reshape(1, -1))

            color = color_map[:, prediction_label].reshape(1, -1)
            colored_img[sub_region[0], sub_region[1], :] = color
    return colored_img


def refine(component):
    """Filter out extremely small components."""
    if component.sum() > 30:
        return True
    return False


def match(img, kernel):
    """Return the activated pixel image."""
    image = img
    output = cv2.filter2D(1-img, -1, kernel)
    x, y = (output == kernel.sum()).nonzero()
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            try:
                image[x[i]-1:x[i]+2, y[i]-1:y[i]+2] = np.ones((3, 3), dtype=np.uint8)
            except:
                continue
    return image


def active_regions(skeleton, bank):
    """Find the active regions in the image."""
    img = skeleton
    for filt in bank:
        img = match(img, filt)
    return img


def skeletonize(img):
    """Return 1 pixel thick skeleton of binary image."""
    skeleton = skeletonize_3d(1-img)
    skeleton = 1 - cv2.threshold(skeleton, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return skeleton


def main(input_tuple):
    """Main."""
    root_folder, img_name, cluster, color_map, bank = input_tuple
    assert img_name[-4:] in [".png", ".jpg"]
    print("Processing ", img_name)
    INPUT_FOLDER = "/home/chrizandr/data/Telugu/skel_data/"
    OUTPUT_FOLDER = "/home/chrizandr/data/Telugu/clr_data/"

    img = cv2.imread(root_folder + img_name, 0)
    assert img is not None

    binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    skeleton = skeletonize(binary_img)
    activated = active_regions(skeleton, bank)
    color_image = colorize_strokes(activated, cluster, color_map)

    cv2.imwrite(INPUT_FOLDER + img_name, 255 - (skeleton*255))
    cv2.imwrite(OUTPUT_FOLDER + img_name, 255-color_image)

    return None


if __name__ == "__main__":
    CLUSTER = pickle.load(open("cluster.pkl", "rb"))
    BANK = pickle.load(open("../../banks/Py2.7/J34_3.pkl", "rb"))
    COLOR_MAP = get_color_map(len(CLUSTER.cluster_centers_))
    DATA_FOLDER = "/home/chrizandr/data/Telugu/handwritten/"

    folderlist = os.listdir(DATA_FOLDER)
    folderlist.sort()

    input_list = [(DATA_FOLDER, x, CLUSTER, COLOR_MAP, BANK) for x in folderlist]

    pool = multiprocessing.Pool(6)
    result = pool.map(main, input_list)
