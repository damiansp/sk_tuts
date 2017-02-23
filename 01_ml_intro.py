from sklearn import datasets, random_projection, svm
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
import pickle

iris = datasets.load_iris()
digits = datasets.load_digits()

print digits.data
print digits.target
print digits.images[0]

clf = svm.SVC(gamma = 0.001, C = 100.)

clf.fit(digits.data[:-1], digits.target[:-1])

print clf.predict(digits.data[-1:])



clf2 = svm.SVC()
X, y = iris.data, iris.target

clf2.fit(X, y)

# Save model
s = pickle.dumps(clf2)

# Load
clf2 = pickle.loads(s)
print clf2.predict(X[0:1])
print y[0]

# More efficient storage and loading (to disk only):
joblib.dump(clf, 'testfile.pkl')

# Reload
clf = joblib.load('testfile.pkl')


# Sklearn defaults to float64
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype = 'float32')
print X.dtype

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print X_new.dtype

# Regression targets are cast to float64; classification targets are maintained
clf = svm.SVC()
clf.fit(iris.data, iris.target)

print list(clf.predict(iris.data[:3]))
clf.fit(iris.data, iris.target_names[iris.target])
print list(clf.predict(iris.data[:3]))



# Refitting and updating params
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

clf = svm.SVC()
clf.set_params(kernel = 'linear').fit(X, y)
print clf.predict(X_test)

clf.set_params(kernel = 'rbf').fit(X, y)
print clf.predict(X_test)



# Multiclass vs multilabel fitting
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0 , 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator = svm.SVC(random_state = 0))
print classif.fit(X, y).predict(X)

y = LabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)

y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print classif.fit(X, y).predict(X)
