a"""Creating text blocks from the text image."""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity


def get_ids(name):
    """Get the ID of the writers."""
    f = open(name, "r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = int(line[1])
    return dictionary


def refine_by_size(components, shape):
    """Remove useless components."""
    useful = list()
    for component in components:
        # Binarising
        struct = np.ones((3, 3), dtype=np.uint8)
        # Eroding and Dilating
        img = cv2.dilate(1-component, struct, iterations=1)
        img = cv2.erode(img, struct, iterations=1)
        # Thresholding
        high = (img.shape[0]*img.shape[1])*0.75
        isum = (img).sum()
        if component.shape[0] < shape and component.shape[1] < shape:
            if isum < high:
                useful.append(component)
    return useful


def get_connected_components(img):
    """Get the connected components."""
    print("Getting the connectedComponents")
    nimg = cv2.connectedComponents(1-img)[1]

    labels = set(np.unique(nimg))
    labels.remove(0)
    components = list()

    for label in labels:
        sub_region = (nimg == label).nonzero()
        max_hor = sub_region[1].max()
        min_hor = sub_region[1].min()
        max_ver = sub_region[0].max()
        min_ver = sub_region[0].min()
        region = img[min_ver:max_ver, min_hor:max_hor]
        if max_hor - min_hor > 3 and max_ver - min_ver > 3:
            components.append(region)
    return components


def strip_white(img):
    """Remove white parts surrounding the image."""
    m, n = img.shape
    proj = np.sum(img, axis=1)
    cords = (proj != (255*n)).nonzero()
    return img[cords[0][0]:cords[0][-1]+1, :]


def match(img, kernel):
    """Count areas where kernel matches images."""
    output = cv2.filter2D(img, -1, kernel)
    return len((output[1:-1, 1:-1] == kernel.sum()).nonzero()[0])


def feature_LBP(img):
    """LBP feature for the image."""
    m = img.shape[0]
    n = img.shape[1]
    img = local_binary_pattern(img, 8, 1, 'default')
    # pdb.set_trace()
    img = img.astype(np.uint8)
    feature = [0 for i in range(0, 256)]
    for i in range(0, m):
        for j in range(0, n):
            feature[img[i, j]] += 1
    return feature


def refine_texture(components, img):
    """Remove components that have noisy textures."""
    # Finding global feature image
    f = np.array(feature_LBP(img))

    # Finding feature for each of the connectedComponents
    feature = list()
    for comp in components:
        feature.append(feature_LBP(comp))
    feature = np.array(feature)

    # Calculating cosine_similarity with the global feature
    similarity = list()
    for i in range(feature.shape[0]):
        similarity.append((cosine_similarity(feature[i].reshape((1, -1)), f.reshape((1, -1)))[0][0], components[i]))

    # Sorting based on similarity
    similarity = sorted(similarity, key=lambda tup: tup[0], reverse=True)
    # Separating similarity and components
    values = [x[0] for x in similarity]
    comps = [x[1] for x in similarity]
    out = list()
    # Thresholding based on similarity

    index = (np.array(values) > 0.4).nonzero()[0][-1] + 1
    out.append(comps[0:index])

    return out
