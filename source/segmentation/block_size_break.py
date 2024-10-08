"""Fixed sized block construction with disconnected components."""

import cv2
# import pdb
import numpy as np
import os
import multiprocessing
from scipy import ndimage
from block import strip_white, get_connected_components, refine_by_size, refine_texture


def projection(a, flag):
    """Projection of matrix to a vector."""
    nrows, ncols = a.shape
    ver_filter = np.ones((ncols, 1))
    hor_filter = np.ones((1, nrows))
    if flag == 1:
        proj = np.dot(a, ver_filter)
        proj = proj.T
        # Dimensions : 1 x nrows
    elif flag == 2:
        proj = np.dot(hor_filter, a)
        # Dimensions : 1 x ncols
    return proj


def construct_block(components, max_height, shape):
    """Create block from connected components."""
    blocks = list()
    for each in components:
        cmy, cmx = tuple([int(x) for x in ndimage.measurements.center_of_mass(each)])
        block = np.ones((max_height, each.shape[1])) * 255
        block[max_height/2 - cmy:each.shape[0] - cmy + max_height/2, :] = each
        blocks.append(block)
    for i in range(1, len(blocks)):
        blocks[0] = np.concatenate((blocks[0], blocks[i]), axis=1)
    return strip_white(blocks[0])


def get_blocks(base, shape):
    """Get blocks from image."""
    blocks = list()
    lines = list()
    height, width = shape
    index = 0
    while index + width < base.shape[1]:
        lines.append(base[:, index:index+width])
        index += width
    lines = [strip_white(x) for x in lines]
    block = lines[0]
    for i in range(1, len(lines)):
        block = np.concatenate((block, lines[i]), axis=0)
    index = 0
    while index + height < block.shape[0]:
        blocks.append(block[index:index+height, :])
        index += height
    return blocks


def extract(name):
    """Extract blocks from an image."""
    writers = list()
    if name[-4:] == ".png":          # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        print("Processing " + name)
        img = cv2.imread("/home/chrizandr/data/telugu_hand2/" + name, 0)
        b_img = 1 - cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = 255 - (b_img * (255 - img))
        components = get_connected_components(img)
        components = refine_by_size(components, 300)
        components = refine_texture(components, b_img)
        baseTexture = construct_block(components, img.shape[0])
        blocks = get_blocks(baseTexture, (300, 300))
        count = 0
        print("Writing blocks")
        for block in blocks:
            cv2.imwrite("/home/chris/data/temp/" + name[0:-4] + '-' + str(count) + ".png", block)
            count += 1
    return writers


if __name__ == "__main__":
    data_path = "/home/chrizandr/data/telugu_hand2/"              # Path of the original dataset
    output_path = "/home/chrizandr/data/temp/"            # Path of the output folder
    folderlist = os.listdir(data_path)
    # folderlist.remove("writerids.csv")
    folderlist.sort()

    pool = multiprocessing.Pool(6)
    result = pool.map(extract, folderlist)
