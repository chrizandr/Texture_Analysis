# -------------------------------------------------
import cv2
import numpy as np
# -------------------------------------------------

# ------------------------------ Global variables for Gabor features.
sigma = 2                       # Variance of the Gaussian Kernel
frequencies = [4,8,16,32]       # The scales of the different Gabor filters to be used
angles = [0,45,90,135]          # The orientations of the different Gabor filters to be used
# ------------------------------

def feature_set(name,frequencies,angles,sigma):
    img = cv2.imread(name,0)
    ret, img = cv2.threshold(img , 0 , 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features=[]
    for f in frequencies:
        for theta in angles:
            x=get_Gabor_features(img,f,theta,sigma)
            a=x.mean()*100
            b=x.std()*100
            features.append(a)
            features.append(b)
    return features

def get_Gabor_features(img,f,theta,sigma):
    theta = getangle(theta)
    gaussian_kernel = make_gaussian_kernel((3,3),sigma)
    Gka, Gkb = make_Gabor_kernel(gaussian_kernel,f,theta)
    img1 = cv2.filter2D(img,-1,Gka)
    img2 = cv2.filter2D(img,-1,Gkb)
    img1 = img1*img1
    img2 = img2*img2
    x = img1+img2
    feature = np.sqrt(x)
    return feature

def make_Gabor_kernel(gkernel,f,theta):
    he=np.arange(gkernel.size).reshape(gkernel.shape)
    ho=np.arange(gkernel.size).reshape(gkernel.shape)
    he=he.astype(np.float64)
    ho=ho.astype(np.float64)
    for i in range(he.shape[0]):
        for j in range(he.shape[1]):
            val= 2*np.pi*f*((i+1)*np.cos(theta)+(j+1)*np.sin(theta))
            he[i,j]=gkernel[i,j]*np.cos(val)
    for i in range(ho.shape[0]):
        for j in range(ho.shape[1]):
            val= 2*np.pi*f*((i+1)*np.cos(theta)+(j+1)*np.sin(theta))
            ho[i,j]=gkernel[i,j]*np.sin(val)
    return he,ho

def make_gaussian_kernel(ksize, sigma):
    kx=cv2.getGaussianKernel(ksize[0],sigma);
    ky=cv2.getGaussianKernel(ksize[1],sigma);
    kernel = kx*ky.transpose()
    return kernel

def getangle(n):
    return (np.float64)(n*np.pi)/180

# ---------------------------------------------------------------------------------------------------------
