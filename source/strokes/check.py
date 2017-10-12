#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import os
import pdb
import cv2
from xy_fea import feature
import operator
import numpy as np

cluster = pickle.load(open('cluster.pkl', 'rb'))
path = '/home/chrizandr/data/Telugu/strokes/'
data = os.listdir(path)

cluster_no = cluster.cluster_centers_.shape[0]

stroke_dic = dict()
for i in range(cluster_no):
    stroke_dic[i] = 0

for each in data:
    f_path = path + each + '/'
    folderlist = os.listdir(f_path)
    folderlist.sort()
    print("Checking ", each)
    for name in folderlist:
        try:
            img_name = f_path + name
            img = cv2.imread(img_name, 0)[1:-1, 1:-1]
        except TypeError:
            pdb.set_trace()
        feat = feature(img, 25, 25)
        max_v = max(feat)
        feat = np.array([float(x) / max_v for x in feat]).reshape(1, -1)
        stroke_dic[cluster.predict(feat)[0]] += 1

max_val = max(stroke_dic.items(), key=operator.itemgetter(1))[1]
rank = 0

# pdb.set_trace()
f = open("ziph.dump", "w")
print("Writing")
while len(stroke_dic) > 0:
    key, max_val = max(stroke_dic.items(), key=operator.itemgetter(1))
    del stroke_dic[key]
    rank += 1
    f.write("Rank: " + str(rank) + ", Cluster no: " + str(key) + ", freq: " + str(max_val) + ", zipf freq: " + str(max_val/rank) + '\n')

f.close()
pdb.set_trace()
