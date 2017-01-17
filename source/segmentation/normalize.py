# Used to normalize a given image from the IAM dataset. The images must be form images.
# Dependencies : OpenCV, numpy,
# ------------------------------------------
import cv2
import numpy as np
import os
import pdb
# ------------------------------------------
import segment
# ------------------------------------------
def normalize(img):
	# -------------------------
    # Convert the image into a binary image using Otsu thresholding
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Segment the handwritten part from the image
    seg_img = segment.segment_text(img)
    # Reconstruct the image to remove variable inter-line spacing.
    # Constant spacing of 5 pixels used
    rc_img = segment.reconstruct(seg_img)
    # -------------------------
    return rc_img
# ------------------------------------------
# Optional main function
data_path = "/home/chris/honours/fullimg/"              # Path of the original dataset
output_path = "/home/chris/honours/seg_img/"            # Path of the output folder
# Get a list of all the files in the dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.sort()
log = open("normalize.log","w")
# Open the output file in write mode
print("Starting........")
# img = cv2.imread("test.png",0)
# normal_img = normalize(img) * 255
# cv2.imwrite("result.png",normal_img)
for name in folderlist:
    if name[-4:]==".png":          # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        try:
            print("Processing "+ name)
            img = cv2.imread(data_path + name , 0)
            n_img = normalize(img) * 255
            x = cv2.imwrite(output_path + name,n_img)
            print(x)
        except:
            print("Error at :" + name)
            log.write(name+'\n')
log.close()
