import csv
import pdb


def getIds(id_file):
    """Get the ids of the writers."""
    f = open(id_file, "r")
    dictionary = dict()
    for line in f:
        l = line.strip()
        l = l.split(",")
        dictionary[l[0]] = l[1]
    return dictionary


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='>'):
    """
    Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    reader = csv.reader(open("before.csv", "rb"), delimiter=',')
    data = list()
    classes = list()
    for row in reader:
        data.append(row[0:-1])
        classes.append(row[-1])
    ids = getIds("/home/chrizandr/data/writerids.csv")
    classes = [c + '.png' for c in classes]
    classes = [ids[c] for c in classes]
    f = open("after1.csv", "wb")
    for i in range(len(data)):
        for d in data[i]:
            f.write(d+'\t')
        f.write(classes[i]+'\n')
    f.close()
