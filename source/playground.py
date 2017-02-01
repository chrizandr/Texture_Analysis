import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.png',0)

edges = cv2.Canny(img, 0 ,255)

plt.imshow(255 - edges,'gray')
plt.show()


def get(writers , ids , files , n):
    result = list()
    for writer in writers:
        count = 0
        for i in range(len(ids)):
            if ids[i] == writer:
                count +=1
                result.append(files[i])
            if count == n:
                break
    return result

count = dict()
for i in ids:
    if i in count:
        count[i] +=1
    else:
        count[i] = 1
s = 0
for i in count.iterkeys():
    s += count[i]
cw = list()
for writer in writers:
    if count[writer] > 3:
        cw.append(writer)

for a in ans:
    copyfile("/home/chris/honours/IAM_block/"+a+".png" , "/home/chris/honours/Testing/"+a+".png")
