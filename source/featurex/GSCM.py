import numpy as np
from skimage import feature,io
from skimage.filters import threshold_otsu
import cv2
# ------------------------------ Global variables for GSCM features.
distances = [1,2,3,4,5]         # The distances of the various GSCM matrices to be computed
angles = [0,45,90,135]          # The orientations of the various GSCM matrices to be computed
# ------------------------------

def getangle(n):
    return (np.float64)(n*np.pi)/180

def feature_set(name,distances,angles):
    img = cv2.imread(name,0)
    import pdb; pdb.set_trace()
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for i in range(len(angles)):
        angles[i] = getangle(angles[i])
    features = list()
    x = feature.greycomatrix(img,distances,angles,2)
    import pdb; pdb.set_trace()
    for d in range(len(distances)):
        for theta in range(len(angles)):
            features.append(x[0,0,d,theta])
            features.append(x[1,0,d,theta])
            features.append(x[1,1,d,theta])
    return features
feature_set("test.png",distances,angles)
