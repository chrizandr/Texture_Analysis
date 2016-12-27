import numpy as np
from sklearn.svm import SVC
import pdb

f = open("/home/chris/honours/reduced.txt","r")
writers = list()
for line in f:
    writers.append(int(line))

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

def load_data(filename):
    global writers
    f = open(filename , 'r')
    data = list()
    filename = list()
    ids = get_ids("/home/chris/honours/text_blocks/writerids.csv")
    for line in f:
        l = line.strip()
        l = l.split(',')
        if int(ids[l[-1]]) in writers:
            data.append([float(x) for x in l[0:-1]])
            filename.append(l[-1])
    return np.array(data) , filename

def divide_data(data , tags):
    

def classify(filename , writers):
    data , tags = load_data(filename)
    test , train = divide_data(data , tags)
    files = np.loadtxt(open("/home/chris/honours/Texture_Analysis/writerids.csv" , "r"), delimiter = ',')
    pdb.set_trace()


# data_dir1 = "/home/chris/honours/Texture_Analysis/data_fullimg_csv/"
data_dir2 = "/home/chris/honours/Texture_Analysis/data_block_csv/"
names1 = ["GSCM/GSCM_1","GSCM/GSCM_2","GSCM/GSCM_3","GSCM/GSCM_4","GSCM/GSCM_5","GSCM/GSCM_all","GSCM/GSCM_1_LDA","GSCM/GSCM_2_LDA","GSCM/GSCM_3_LDA","GSCM/GSCM_4_LDA","GSCM/GSCM_5_LDA","GSCM/GSCM_all_LDA"]
names2 = ["Gabor/Gabor_all", "Gabor/Gabor_4", "Gabor/Gabor_8", "Gabor/Gabor_16", "Gabor/Gabor_32", "Gabor/Gabor_all_LDA", "Gabor/Gabor_4_LDA", "Gabor/Gabor_8_LDA", "Gabor/Gabor_16_LDA", "Gabor/Gabor_32_LDA", ]
names3 = ["Edge/Edge_all","Edge/Edge_8","Edge/Edge_12","Edge/Edge_16","Edge/Edge_dp_16","Edge/Edge_all_LDA","Edge/Edge_8_LDA","Edge/Edge_12_LDA","Edge/Edge_16_LDA","Edge/Edge_dp_16_LDA",]

results = list()
for data_dir in [data_dir2]:
    for filename in names2:
        print("Classifying : " + filename)
        # clsf.options = ['-K', '1', '-W', '0' , '-A' ,'weka.core.neighboursearch.KDTree -A "weka.core.EuclideanDistance -R first-last" -S weka.core.neighboursearch.kdtrees.SlidingMidPointOfWidestSide -W 0.01 -L 40 -N']
        evl = classify(data_dir+"Gabor/Gabor_4"+".csv" , writers)
        print("Evaluating")
        # ------------------------------------
        results.append((filename.split('/')[1], evl))
pdb.set_trace()
