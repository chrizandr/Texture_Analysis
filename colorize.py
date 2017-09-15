"""Extract strokes from a given image."""

import cv2
from skimage.morphology import skeletonize_3d
import matplotlib.pyplot as plt
import pickle
import pdb
import numpy as np
import os
import multiprocessing
import pickle
import random
from xy_fea import *
import scipy.misc

def get_color_map(n):
    random.seed()
    colormap = list()
    colormap_img = np.ones((200,200,3))
    for i in range(n):
        a = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        while a not in colormap:
            colormap.append(a)
    i=0
    for each in colormap:
        colormap_img[i:i+3,:,:] = each
        i+=3
    #plt.imshow(colormap_img)
    #plt.show()
    return np.array(colormap)

cluster = pickle.load(open("cluster.p","rb"))
color_map = get_color_map(len(cluster.cluster_centers_))


def colorize_strokes(activated):
    """Colorize strokes based on clustering"""
    nimg = cv2.connectedComponents(1-activated)[1]
    colored_img = np.ones((nimg.shape[0], nimg.shape[1], 3))
    # Get the connectedComponents
    labels = set(np.unique(nimg))
    labels.remove(0)
    components = list()
    # Extracting the components
    #cluster = pickle.load(open("cluster.p","rb"))
    #color_map = get_color_map(len(cluster.cluster_centers_))
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
                img = (img*255)[1:-1, 1:-1]
                fea = feature(img, 20, 21)
                #plt.imshow(img, cmap='gray')
                #plt.show()
                #pdb.set_trace()
                color = color_map[cluster.predict(np.array(fea).reshape(1,-1))]
                colored_img[sub_region[0],sub_region[1],:] = color
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


def active_regions(skeleton):
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


def extract_strokes(img_name):
    """Extract strokes from a given image."""
    if img_name[-4:] == ".png":
        img = cv2.imread("/home/chrizandr/data/writing_segment/" + img_name, 0)
        binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        skeleton = skeletonize(binary_img)
        folder = "/home/chrizandr/data/skel_data/"
        scipy.misc.toimage(255*skeleton).save(folder+img_name)
        activated = active_regions(skeleton)
        clr_img = colorize_strokes(activated)
        folder2 = "/home/chrizandr/data/clr_data/"
        scipy.misc.toimage(255-clr_img, channel_axis=2).save(folder2+img_name)  


bank = pickle.load(open("banks/Py2.7/J34_3.pkl", "rb"))
data = "/home/chrizandr/data/writing_segment/"
folderlist = os.listdir(data)
folderlist.sort()

#extract_strokes("a-0.png")
pool = multiprocessing.Pool(6)
result = pool.map(extract_strokes, folderlist)

#pdb.set_trace()
