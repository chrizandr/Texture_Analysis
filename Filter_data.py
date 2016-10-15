import pdb
import os

writer_file = open("writerids.csv",'r')
writer_count = dict()
filename = list()
writername = list()
for i in range(0,700):
    writer_count[i] = 0
for line in writer_file:
    line = line.strip()
    line = line.split(',')
    filename.append(line[0])
    writername.append(int(line[1]))
    writer_count[int(line[1])] += 1


for i in range(len(filename)):
     if writer_count[int(writername[i])] < 2:
         os.remove("fullimg_norm/"+filename[i]+".png")
