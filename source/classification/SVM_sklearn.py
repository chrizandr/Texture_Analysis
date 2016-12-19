from sklearn import svm,cross_validation,preprocessing
from sklearn.pipeline import make_pipeline
import sys

path = "data_fullimg_csv/Edge_direction_fullimg/"
train_file = open(path+sys.argv[1]+".csv","r")
train_data=[]
train_class=[]
for line in train_file:
    l = line.strip()
    l = l.split(',')
    l = map(float , l)
    train_data.append(l[0:-1])
    train_class.append(int(l[-1]))
clf = svm.SVC()
n_samples = len(train_data)
cv = cross_validation.ShuffleSplit(n_samples, n_iter=10,test_size=0.3, random_state=0)
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC())
print cross_validation.cross_val_score(clf, train_data, train_class, cv=cv)
