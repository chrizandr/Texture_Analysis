import pdb

f= open("writerid_eng.csv", 'r')
dic = dict()
count = 0
for line in f:
    line = line.strip().split(',')
    if line[1] in dic:
       dic[line[1]].append(line[0])
    else:
       dic[line[1]] = [line[0]]

required_files = []
for each in dic.keys():
    if len(dic[each])>2:
       count+=1
       for item in dic[each]:
           required_files.append(item)

f2 = open("valid.csv",'w')
for each in required_files:
    f2.write(each+'\n')
f2.close()
f.close()
