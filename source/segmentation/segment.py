# Fucntions used to help in the segmentation and reconstruction process of the image
# --------------------------------------------
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pdb
# --------------------------------------------
def segment_text(img):
    # ------------------------------------
    padding = 50
    neigbourhood = 100
    # Cutting out the printed region. Vertical projections of mid-point neigbourhood has local minima
    img = img[:,padding:img.shape[1]-padding]
    mid = int(img.shape[1]/2)
    proj = projection(img[:,mid-neigbourhood:mid+neigbourhood],1)
    start,end = cluster3((proj==0).nonzero()[1])
    seg_img = img[start:end,:]
    # ---------------------
    # Removing the remainder white background and returning only the text block
    threshold = 20
    max_ver, max_hor = seg_img.shape
    hor_points = (projection(seg_img,2) <= max_ver-threshold).nonzero()[1]
    ver_points = (projection(seg_img,1) <= max_hor-threshold).nonzero()[1]
    # ------------------------------------
    return seg_img[ver_points[0]-padding:ver_points[-1]+padding,hor_points[0]:hor_points[-1]]

def reconstruct(img):
    # ------------------------------------
    # Reconstruct the image removing the variable inter-line spacing and replace with constant spacing
    padding = 5             # The constant spacing
    threshold = 20          # The minimum possible width of a text line
    # -----------------------
    # Finding all the white lines
    points = (projection(img,1)==img.shape[1]).nonzero()[1]
    # Comuting the gradient of the indices and collecting all points where difference is more than threshold
    grad = [(points[x],points[x+1]) for x in (gradient(points)>threshold).nonzero()[0]]
    # -----------------------
    # Reconstructing the image using the text lines and constant padding
    rs_img = img[grad[0][0]:grad[0][1],:]
    for i in range(1,len(grad)):
        rs_img = np.concatenate( (rs_img , np.ones((padding,img.shape[1])) , img[grad[i][0]:grad[i][1],:] ) , 0)
    # ------------------------------------
    return rs_img

def projection(a,flag):
    # ------------------------------------
    # Projecting the image on the horizontal or vertical axis by multiplying with vector of 1's
    nrows,ncols = a.shape
    ver_filter = np.ones((ncols,1))
    hor_filter = np.ones((1,nrows))
    # ----------------------
    # vertical projection, the array needs to be transposed to make it into a vector
    if flag==1 :
        proj = np.dot(a,ver_filter)
        proj = proj.T
        # Dimensions : 1 x nrows
    # ----------------------
    # horizontal projection, the array is already in the for of a vector
    elif flag==2 :
        proj = np.dot(hor_filter,a)
        # Dimensions : 1 x ncols
    # ------------------------------------
    return proj

def gradient(vector):
    # ------------------------------------
    # Computing the gradient for the given set of points
    grad = list()       # List containing the final gradient values
    # Store the previous value in the set
    prev = vector[0]
    for i in range(len(vector)-1):
        grad.append(abs(vector[i+1] - vector[i]))
    # Convert the list into a vector
    grad = np.array(grad)
    # ------------------------------------
    return grad

def cluster3(array):
    # ------------------------------------
    # Need to cluster the given set of values into three regions and return the mid region co-ordinates
    threshold = 50          # The minimum distance between the regions
    padding = 20            # Extra padding for the middle region
    # Calculating the gradient of points and finding all points where difference is greater than threshold
    cords = (gradient(array)>threshold).nonzero()[0]
    # Storing the co-ordinates for the middle region
    start_col = array[cords[0]+1] + padding
    end_col = array[cords[1]+1] - padding
    # ------------------------------------
    return start_col,end_col

# --------------------------------------------


data_path = "/home/chris/honours/fullimg/"              # Path of the original dataset
output_path = "/home/chris/honours/hand_img/"            # Path of the output folder
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
        # try:
        print("Processing "+ name)
        img = cv2.imread(data_path + name , 0)
        img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        nimg = segment_text(img)
        # pdb.set_trace()
        filename = output_path + name
        nimg = nimg * 255
        x = cv2.imwrite(filename , nimg)

        # except:
        #     print("Error at :" + name)
        #     log.write(name+'\n')
log.close()
