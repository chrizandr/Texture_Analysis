import cv2
import pdb
import os
import matplotlib.pyplot as plt

def extract(img_name, folder):
    OUTPUT_FOLDER = "/home/chrizandr/data/firemaker/linear_color_maps/"
    img = cv2.imread(folder + img_name)
    nimg = img[750:3300, 50:-50, :]
    x = cv2.imwrite(OUTPUT_FOLDER + img_name, nimg)
    print(x)
    return None


if __name__ == "__main__":
    FOLDER = "/home/chrizandr/data/firemaker/linear_wrong/"
    names = os.listdir(FOLDER)
    # extract("15701.tif", FOLDER)
    for name in names:
        print(name)
        extract(name, FOLDER)
