# -------------------------------------- [Could measure the time taken to process each file if needed]
# import time
# Uncomment the lines to measure and print the time
# --------------------------------------

# **************************************************
# Import the source for the desired features from the feature source file. Use below given format.
from Gabor import *
import os
import time
# NOTE: Please import only one feature source. Multiple source import will confuse the compiler.
# **************************************************

def get_ids(id_file):
    f = open(id_file,"r")
    dictionary = dict()
    for line in f:
        line = line.split(",")
        dictionary[line[0]] = line[1]
    return dictionary

# **************************************************


# **************************************************

data_path = "/home/chris/honours/text_blocks/"
output_file ="output.csv"
# Give the path of the file conataining the class labels for the images.
# Format for the file [each new line conataining] in '.csv' format
# <filename[without the file extension]>,<classlabel>
class_labels = "/home/chris/honours/text_blocks/writerids.csv"
# Construct a dictionary out of the given file
labels = get_ids(class_labels)
# Get a list of all the files in teh dataset folder [data_path] and sort them alphabetically
folderlist = os.listdir(data_path)
folderlist.sort()
# Open the output file and the log file in write mode
f = open(output_file,"w")
log = open("featurex.log","w")
# Loop over the files in the dataset
for name in folderlist:
    if name[-4:]=='.png':     # Make sure that only appropriate files are processed [add 'or' conditions for other filetypes]
        try:
            print("Processing "+ name)

            label = labels[name[0:-4]]

            start_time = time.time()          # Code for time measurement
            img_name = data_path + name
            # ------------------------------ The feature set function from the appropriately imported feature source is used.
            A = feature_set(img_name,frequencies,angles,sigma)
            # ------------------------------
            for feature in A:
                f.write(str(feature)+',')
            f.write(label)
            f.write("\n")
            print("--- %s seconds ---" % (time.time() - start_time))      # Code for time measurement
        except:                         # Log any files that have some error
            log.write(name+'\n')
log.close()
f.close()
