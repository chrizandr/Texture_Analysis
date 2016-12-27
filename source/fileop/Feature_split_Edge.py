import pdb
path = "/home/chris/honours/Texture_Analysis/data_block_csv/Edge/"
train_file = open(path+ "output.csv","r")
train_data = []
train_class = []
for line in train_file:
	k=line.strip()
	k=k.split(',')
	ky = map(float,k[0:-1])
	train_data.append(ky)
	train_class.append(k[-1])
train_file.close()
Edge_8 = []
Edge_12 = []
Edge_16 = []
Edge_dp_16 = []

for i in range(len(train_data)):
	Edge_8.append(train_data[i][0:8])
	Edge_12.append(train_data[i][8:20])
	Edge_16.append(train_data[i][20:36])
	Edge_dp_16.append(train_data[i][36:51])
	Edge_8[i].append(train_class[i])
	Edge_12[i].append(train_class[i])
	Edge_16[i].append(train_class[i])
	Edge_dp_16[i].append(train_class[i])

data_list =[]
data_list.append(Edge_8)
data_list.append(Edge_12)
data_list.append(Edge_16)
data_list.append(Edge_dp_16)

file_list=["Edge_8","Edge_12","Edge_16","Edge_dp_16"]
for i in range(4):
	f = open(path+file_list[i]+".csv","w")
	data = data_list[i]
	for x in data:
		for k in range(len(x)):
			if k!=len(x)-1:
				f.write(str(x[k])+',')
			else:
				f.write(str(x[k])+'\n')
	f.close()
f = open("Edge_all.csv","w")
for i in range(len(train_data)):
	for data in train_data[i]:
		f.write(str(data)+',')
	f.write(str(train_class[i])+'\n')
f.close()
