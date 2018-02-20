"""SVM Voting classifier."""
################################################################
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from random import shuffle
from collections import Counter
import csv
import pdb
################################################################
# ---------------------------------------------------------------

'''
Returns dictionary mapping from page to writer id
'''


def getIds(filename):
    """Get the writer ids."""
    ids = dict()
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ids[row[0][0:-4]] = int(row[1])
    return ids
# ---------------------------------------------------------------


'''
Splits the data into three parts
Training data, training labels
Test_data ---> dictionary of test pages containing all blocks for each page
'''


def dataSplit(data, labels, ids):
    """Split the data into train and test."""
    X = np.array(data, dtype=np.float)

    #pages = [i.split('-')[0]+'-'+i.split('-')[1] for i in labels]
    #y = np.array([ids[i] for i in pages], dtype=np.int)
    #shuffle(pages)

    test_data = list()
    test_class = list()
    train_data = list()
    train_class = list()

    #for page in set(pages):
    #    if ids[page] not in test_pages:
    #        test_pages.append(ids[page])
    #        test_data[page] = list()
    #pages = [i.split('-')[0]+'-'+i.split('-')[1] for i in labels]
    count = 0
    for i in range(X.shape[0]):
        class_id = ids[labels[i]]
        #pdb.set_trace()
        print(class_id, labels[i])
        if class_id not in test_data:
            test_data.append(X[i])
            test_class.append(class_id)
            count += 1
        else:
            train_data.append(X[i])
            train_class.append(class_id)
    #for page in test_data:
    #    test_data[page] = np.array(test_data[page], dtype=np.float)

    return np.array(train_data, dtype=np.float), np.array(train_class, dtype=np.int), np.array(test_data, dtype=np.float), np.array(test_class, dtype=np.int)


# ---------------------------------------------------------------


'''
Classifies a given feature file using a classifier
'''


def classify(filename, folds):
    """Classification."""
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        data = list()
        labels = list()
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
    ids = getIds("/home/chrizandr/data/Telugu/writerids.csv")
    accuracies = list()
    for i in range(folds):
        print(i)
        train_data, train_class, test_data, test_class = dataSplit(data, labels, ids)
        svm = SVC(kernel='linear')
        svm.fit(train_data, train_class)
        correct = 0.0
        for page in test_data:
            page_class = ids[page]
            output = svm.predict(test_data[page])
            count = Counter(output)
            label = count.most_common()[0][0]
            if label == page_class:
                correct += 1
        accuracy = correct/float(len(test_data))
        accuracies.append(accuracy)

    return 100 * sum(accuracies)/len(accuracies)
# ---------------------------------------------------------------


if __name__ == "__main__":
    data_dir = "/home/chrizandr/Texture_Analysis/source/strokes/"
    names = ["distributions"]
    results = list()
    for filename in names:
        print("Classifying : " + filename)
        evl = classify(data_dir + filename + ".csv", 10)
        print("Evaluating")
        # ------------------------------------
        results.append((filename, evl))
    pdb.set_trace()
