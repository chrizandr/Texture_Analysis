import pdb
import os
def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.strip().split(",")
        dictionary[line[0]] = line[1]
    return dictionary

data_path = "/home/chris/honours/text_blocks_g/"
output_file ="/home/chris/honours/text_blocks_g/writerids.csv"
writers = open(output_file , 'w')
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
class_labels = "/home/chris/honours/Texture_Analysis/writerids.csv"
folderlist = os.listdir(data_path)
folderlist.sort()

labels = get_ids(class_labels)

for name in folderlist:
    for i in range(0,3):
        try:
            label = labels[name[0:-4-i]]
            writers.write(name[0:-4] + ',' + label + '\n')
            break
        except KeyError:
            print("Skip - " + name + " - " +str(i))
