import numpy as np
from sklearn.svm import SVC
import pdb
from random import shuffle
from collections import Counter
# -----------------------------------------------------

def match(tag , names):
    for i in range(0,3):
        try:
            names[tag[0:-i]]
            return tag[0:-i]
        except KeyError:
            continue
    return -1
# -----------------------------------------------------

# Getting the class ids of the writers based on the given csv filename. Generates a dictionary
def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.strip()
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

# -----------------------------------------------------

# Loads the data stored in a csv file and divides it into the data and the class label
def load_data(filename):
    global writers
    f = open(filename , 'r')
    print filename
    data = list()
    names = list()
    for line in f:
        l = line.strip()
        l = l.split(',')
        data.append([float(x) for x in l[0:-1]])
        names.append(l[-1])
    return np.array(data) , names

# -----------------------------------------------------

# Divides the data into training and testing. Divides the pages. Later used to divide the blocks
def divide_data(data , tags):
    files = get_ids("/home/chris/honours/Texture_Analysis/writerids.csv")
    test_tags = list()
    done = list()
    fil = [x for x in files.iterkeys()]
    shuffle(fil)
    for f in fil:
        if (int(files[f]) not in done):
            done.append(int(files[f]))
            test_tags.append(f) # Test pages
    return get_blocks(data , tags , test_tags)

# -----------------------------------------------------

# Separates the blocks into training and testing data
def get_blocks(data , tags , test_tags):
    test_data = dict()
    names = dict()
    train_data = list()
    train_class = list()
    ids = get_ids("/home/chris/honours/Texture_Analysis/writerids.csv")
    for tag in test_tags:
        test_data[tag] = list()

    count = 0
    for i in range(len(tags)):
        tag = tags[i].split('_')[0]
        if tag in test_data:
            test_data[tag].append(data[i])
            count +=1
        else:
            train_data.append(data[i])
            train_class.append(int(ids[tag]))
    return train_data, train_class , test_data

# -----------------------------------------------------

# Clasiification done using a voting based SVM method
def classify(filename):
    data , tags = load_data(filename)
    ids = get_ids("/home/chris/honours/Texture_Analysis/writerids.csv")
    final = list()
    for i in range(10):
        train_data , train_class , test_data = divide_data(data , tags)
        train_data = np.array(train_data)
        train_class = np.array(train_class)
        svm = SVC()
        svm.fit(train_data , train_class)
        correct = 0
        for page in test_data.iterkeys():
            if len(test_data[page]) == 0:
                pdb.set_trace()
            classid = int(ids[page])
            result = svm.predict(np.array(test_data[page]))
            count = Counter(result)
            label = count.most_common()[0][0]
            if label == classid:
                correct += 1
        final.append((100 * float(correct)) / len(test_data))
    return sum(final)/len(final)


# -------------------------------------------------------------------------------__MAIN__---------------------------------------------
# data_dir1 = "/home/chris/honours/Texture_Analysis/data_fullimg_csv/"
data_dir2 = "/home/chris/honours/Texture_Analysis/data_block_csv/"
names1 = ["GSCM/GSCM_1","GSCM/GSCM_2","GSCM/GSCM_3","GSCM/GSCM_4","GSCM/GSCM_5","GSCM/GSCM_all","GSCM/GSCM_1_LDA","GSCM/GSCM_2_LDA","GSCM/GSCM_3_LDA","GSCM/GSCM_4_LDA","GSCM/GSCM_5_LDA","GSCM/GSCM_all_LDA"]
names2 = ["Gabor/Gabor_all", "Gabor/Gabor_4", "Gabor/Gabor_8", "Gabor/Gabor_16", "Gabor/Gabor_32", "Gabor/Gabor_all_LDA", "Gabor/Gabor_4_LDA", "Gabor/Gabor_8_LDA", "Gabor/Gabor_16_LDA", "Gabor/Gabor_32_LDA", ]
names3 = ["Edge/Edge_all","Edge/Edge_8","Edge/Edge_12","Edge/Edge_16","Edge/Edge_dp_16","Edge/Edge_all_LDA","Edge/Edge_8_LDA","Edge/Edge_12_LDA","Edge/Edge_16_LDA","Edge/Edge_dp_16_LDA",]
names4 = ["LBP/LBP" , "LBP/LBP_LDA"]
results = list()
for data_dir in [data_dir2]:
    for filename in names1+names2+names4:
        print("Classifying : " + filename)
        # clsf.options = ['-K', '1', '-W', '0' , '-A' ,'weka.core.neighboursearch.KDTree -A "weka.core.EuclideanDistance -R first-last" -S weka.core.neighboursearch.kdtrees.SlidingMidPointOfWidestSide -W 0.01 -L 40 -N']
        evl = classify(data_dir+filename+".csv")
        print("Evaluating")
        # ------------------------------------
        results.append((filename.split('/')[1], evl))
pdb.set_trace()
