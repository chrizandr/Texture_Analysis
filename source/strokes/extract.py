"""Extract strokes from a given image."""

import cv2
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
import pickle
import pdb
import numpy as np
import os
import multiprocessing


def extract_strokes(activated, output_image, img_name):
    """Return the connectedComponents of the image."""
    nimg = cv2.connectedComponents(1-activated)[1]
    # Get the connectedComponents
    labels = set(np.unique(nimg))
    labels.remove(0)
    components = list()
    # Extracting the components
    for label in labels:
        sub_region = (nimg == label).nonzero()
        max_hor = sub_region[1].max()
        min_hor = sub_region[1].min()
        max_ver = sub_region[0].max()
        min_ver = sub_region[0].min()
        img = np.ones((max_ver-min_ver+1, max_hor-min_hor+1), dtype=np.uint8)
        img[sub_region[0]-min_ver, sub_region[1]-min_hor] = 0
        if max_hor - min_hor > 5 and max_ver - min_ver > 5:
            if refine(img):
                img = np.vstack((img, np.ones((1, img.shape[1]))))
                img = np.vstack((np.ones((1, img.shape[1])), img))
                img = np.hstack((img, np.ones((img.shape[0], 1))))
                img = np.hstack((np.ones((img.shape[0], 1)), img))
                components.append(img)
    return components


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
    root_folder, img_name, bank = input_tuple
    assert img_name[-4:] in [".png", ".jpg", ".tif"]
    print("Processing ", img_name)
    # INPUT_FOLDER = "/home/chrizandr/data/Telugu/skel_data/"
    OUTPUT_FOLDER = "/home/chrizandr/data/firemaker/strokes/"

    img = cv2.imread(root_folder + img_name, 0)
    assert img is not None

    binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    skeleton = skeletonize(binary_img)
    activated = active_regions(skeleton, bank)
    components = extract_strokes(activated, OUTPUT_FOLDER, img_name)
    folder = OUTPUT_FOLDER + img_name[0:-4] + '/'

    os.makedirs(folder)
    for i in range(len(components)):
        cv2.imwrite(folder + str(i) + ".tiff", components[i] * 255)
    return None


if __name__ == "__main__":
    BANK = pickle.load(open("../../banks/Py3.5/J34_3.pkl", "rb"))
    DATA_FOLDER = "/home/chrizandr/data/firemaker/handwritten/"

    folderlist = os.listdir(DATA_FOLDER)
    folderlist.sort()

    input_list = [(DATA_FOLDER, x, BANK) for x in folderlist]
    # main(input_list[0])
    pool = multiprocessing.Pool(20)
    result = pool.map(main, input_list)
