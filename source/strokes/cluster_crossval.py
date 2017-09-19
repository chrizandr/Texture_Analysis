"""Cross validation for best cluster number."""

import matplotlib.pyplot as plt
import numpy as np
from stroke_cluster import cluster
import operator


def main():
    """Main."""
    print("Getting from file...")
    f = open('features.csv', 'r')
    file_name = []
    fea = []
    for line in f:
        line = line.strip().split(',')
        file_name.append(line[-1])
        fea.append([float(each) for each in line[:-1]])

    print("Converting to array")
    X = np.array(fea)
    lis = []
    for i in range(97, 98):
        print("Custering for k = ", i)
        v = cluster(X, i, file_name)
        print(i, v)
        lis.append(v)

    min_index, min_value = min(enumerate(lis), key=operator.itemgetter(1))
    print min_index, min_value
    plt.plot(range(97, 98), lis)
    plt.xlabel("No. of clusters")
    plt.ylabel("Avg within-cluster variance")
    plt.show()


if __name__ == "__main__":
    main()
