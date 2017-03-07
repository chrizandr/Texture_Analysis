################################################################
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from collections import Counter
import csv, pdb
################################################################
#---------------------------------------------------------------

'''
Returns dictionary mapping from page to writer id
'''

def getIds(filename):
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f , delimiter=',')
        for row in reader:
            ids[ row[0][0:-4] ] = int( row[1] )
    return ids
#---------------------------------------------------------------

'''
Splits the data into three parts
Training data, training labels
Test_data ---> dictionary of test pages containing all blocks for each page
'''

def dataSplit(data, labels, ids):
    X = np.array(data , dtype=np.float)

    pages = [ i.split('-')[0]+'-'+i.split('-')[1] for i in labels]
    y = np.array( [ids[i] for i in pages] , dtype=np.int )
    shuffle(pages)

    test_pages = list()
    test_data = dict()
    train_data = list()
    train_class = list()

    for page in set(pages):
        if ids[page] not in test_pages:
            test_pages.append(ids[page])
            test_data[page] = list()
    pages = [ i.split('-')[0]+'-'+i.split('-')[1] for i in labels]
    count = 0
    for i in range(X.shape[0]):
        if pages[i] in test_data:
            test_data[pages[i]].append(X[i])
            count+=1
        else:
            train_data.append(X[i])
            train_class.append(y[i])
    for page in test_data:
        test_data[page] = np.array(test_data[page], dtype=np.float)

    return np.array(train_data, dtype=np.float), np.array(train_class, dtype=np.int), test_data

#---------------------------------------------------------------

'''
Classifies a given feature file using a classifier
'''

def classify(filename, folds):
    with open(filename) as f:
        reader = csv.reader(f , delimiter=',')
        data = list()
        labels = list()
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
    ids = getIds("/home/chrizandr/data/writerids.csv")
    accuracies = list()
    for i in range(folds):
        print(i)
        train_data, train_class, test_data = dataSplit(data, labels, ids)
        svm = KNeighborsClassifier()
        svm.fit(train_data, train_class)
        correct = 0.0
        for page in test_data:
            page_class = ids[page]
            output = svm.predict(test_data[page])
            count = Counter(output)
            label = count.most_common()[0][0]
            if label==page_class:
                correct += 1
        accuracy = correct/float(len(test_data))
        accuracies.append(accuracy)

    return 100 * sum(accuracies)/len(accuracies)
#---------------------------------------------------------------

# ---------------------------------__MAIN__---------------------------------------------

data_dir = "/home/chrizandr/Texture_Analysis/data_telugu_blocks/"
names = [ "Features/conv_2_3_LDA","Features/conv_2_3","Features/conv_234_3_LDA","Features/conv_234_3","Features/conv_234_4_LDA","Features/conv_234_4"]
names1 = ["Edge_old/Edge_direction","Edge_old/Edge_direction_LDA", "Gabor/Gabor_LDA" , "Gabor/Gabor"]
results = list()
for filename in names1:
    print("Classifying : " + filename)
    evl = classify( data_dir + filename + ".csv", 1)
    print("Evaluating")
    # ------------------------------------
    results.append((filename, evl))
pdb.set_trace()

#---------------------------------------------------------------
