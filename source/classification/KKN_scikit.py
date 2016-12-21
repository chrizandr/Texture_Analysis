import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pdb
f = open("/home/chris/honours/reduced.txt","r")
writers = list()
for line in f:
    writers.append(int(line))


def classify(filename , writers):
    f = open(filename , 'r')
    dataset = np.loadtxt(f,delimiter = ',')
    # dataset = list()
    for entry in raw_dataset:
        if int(entry[-1]) in writers:
            dataset.append(entry)
    dataset = np.array(dataset)
    results = list()
    for model in range(10):
        np.random.shuffle(dataset)
        labels = dataset[:,-1:]
        classes = [int(x) for x in np.unique(labels)]
        train = list()
        test = list()
        for i in range(dataset.shape[0]):
            if int(dataset[i,-1]) in classes:
                test.append(dataset[i,:])
                classes.remove(dataset[i,-1])
            else:
                train.append(dataset[i,:])
        t_train = np.array(train)
        t_test = np.array(test)
        knn = KNeighborsClassifier()
        knn.fit(t_train[:,0:-1],t_train[:,-1])
        result = knn.predict(t_test[:,0:-1])
        correct = 0
        for i in range(result.shape[0]):
            if int(result[i]) == int(t_test[i,-1]):
                correct +=1
        results.append((100*float(correct)) / t_test.shape[0])
    return sum(results)/len(results)
data_dir1 = "/home/chris/honours/Texture_Analysis/data_fullimg_csv/"
data_dir2 = "/home/chris/honours/Texture_Analysis/data_block_csv/"
names1 = ["GSCM/GSCM_1","GSCM/GSCM_2","GSCM/GSCM_3","GSCM/GSCM_4","GSCM/GSCM_5","GSCM/GSCM_all","GSCM/GSCM_1_LDA","GSCM/GSCM_2_LDA","GSCM/GSCM_3_LDA","GSCM/GSCM_4_LDA","GSCM/GSCM_5_LDA","GSCM/GSCM_all_LDA"]
names2 = ["Gabor/Gabor_all", "Gabor/Gabor_4", "Gabor/Gabor_8", "Gabor/Gabor_16", "Gabor/Gabor_32", "Gabor/Gabor_all_LDA", "Gabor/Gabor_4_LDA", "Gabor/Gabor_8_LDA", "Gabor/Gabor_16_LDA", "Gabor/Gabor_32_LDA", ]
names3 = ["Edge/Edge_all","Edge/Edge_8","Edge/Edge_12","Edge/Edge_16","Edge/Edge_dp_16","Edge/Edge_all_LDA","Edge/Edge_8_LDA","Edge/Edge_12_LDA","Edge/Edge_16_LDA","Edge/Edge_dp_16_LDA",]

results = list()
for data_dir in [data_dir1,data_dir2]:
    for filename in names3 + names2 + names1:
        print("Classifying : " + filename)
        # clsf.options = ['-K', '1', '-W', '0' , '-A' ,'weka.core.neighboursearch.KDTree -A "weka.core.EuclideanDistance -R first-last" -S weka.core.neighboursearch.kdtrees.SlidingMidPointOfWidestSide -W 0.01 -L 40 -N']
        evl = classify(data_dir+filename+".csv" , writers)
        print("Evaluating")
        # ------------------------------------
        results.append((filename.split('/')[1], evl))

pdb.set_trace()
