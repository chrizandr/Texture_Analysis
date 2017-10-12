"""Texture based blocks."""
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import pdb
from blocks import strip_white, get_connected_components, refine_by_size
from scipy import ndimage
import multiprocessing


def get_base_texture(comp, max_height, shape):
    """Get the base texture form the image."""
    shuffle_list = np.random.permutation(len(comp))
    components = [comp[x]*255 for x in shuffle_list]
    comps = list()

    for each in components:
        cmy, cmx = tuple([int(x) for x in ndimage.measurements.center_of_mass(each)])
        block = np.ones((max_height, each.shape[1])) * 255
        block[int(max_height/2) - cmy:each.shape[0] - cmy + int(max_height/2), :] = each
        comps.append(block)

    lines = list()
    line = comps[0]
    for i in range(1, len(comps)):
        if line.shape[1] + comps[i].shape[1] >= shape:
            if line.shape[1] > shape:
                line = comps[i]
                print("Omitting component ---------")
            else:
                lines.append(line)
                line = comps[i]
        else:
            line = np.concatenate((line, comps[i]), axis=1)
    lines = [strip_white(np.concatenate((l, 255 * np.ones((l.shape[0], shape - l.shape[1]))), axis=1)) for l in lines]
    blocks = list()
    block = lines[0]
    for i in range(1, len(lines)):
        if block.shape[0] + lines[i].shape[0] > shape:
            blocks.append(block)
            block = lines[i]
            continue
        block = np.concatenate((block, lines[i]), axis=0)
    return blocks


def extract(name):
    """Extact the texture blocks."""
    dataset = "/home/chrizandr/data/Telugu/handwritten/"
    output = "/home/chrizandr/data/Telugu/test/"
    samples = 10
    # Check if the files are images
    if name[-4:] == ".png":
        print("Processing image : "+name)
        img = cv2.imread(dataset + name, 0)
        img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = img[5:-5, 5:-5]
        components = get_connected_components(img)
        components = refine_by_size(components, 224)
        print("Writing blocks")
        for j in range(samples):
            print("Sample number: ", j)
            blocks = get_base_texture(components, img.shape[0], 224)

            for i in range(len(blocks)):
                block = blocks[i].astype(np.uint8)
                block = cv2.cvtColor(block, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(output + name[0:-4] + '-' + str(i) + '-' + str(j) + ".png", block)
        print("Done")


if __name__ == "__main__":
    dataset = "/home/chrizandr/data/Telugu/handwritten/"
    files = os.listdir(dataset)

    # extract(files[0])
    # pdb.set_trace()

    pool = multiprocessing.Pool(5)
    result = pool.map(extract, files)
