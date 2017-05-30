"""Extract strokes from a given image."""

import cv2
from skimage.morphology import skeletonize_3d
# import matplotlib.pyplot as plt
import pickle
import pdb
import numpy as np


def get_connected_components(activated):
    """Return the connectedComponents of the image."""
    nimg = cv2.connectedComponents(1-activated)[1]
    # Get the connectedComponents
    print("Getting the connectedComponents")
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


def active_regions(skeleton):
    """Find the active regions in the image."""
    img = skeleton
    for filt in bank:
        print("Processing bank")
        img = match(img, filt)
    return img


def skeletonize(img):
    """Return 1 pixel thick skeleton of binary image."""
    skeleton = skeletonize_3d(1-img)
    skeleton = 1 - cv2.threshold(skeleton, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return skeleton


def extract_strokes(img):
    """Extract strokes from a given image."""
    binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    skeleton = skeletonize(binary_img)
    activated = active_regions(skeleton)
    components = get_connected_components(activated)
    return components


bank = pickle.load(open("banks/Py2.7/J34_3.pkl", "rb"))
img = cv2.imread("test1.png", 0)
strokes = extract_strokes(img)
print("Writing Stokes")
for i in range(len(strokes)):
    cv2.imwrite("output/" + str(i) + ".tiff", strokes[i] * 255)
pdb.set_trace()
