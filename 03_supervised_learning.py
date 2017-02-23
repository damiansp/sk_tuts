from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(np.unique(iris_y))


# KNN-----------------------------------------------------------------
np.random.seed(9)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# Create and fit KNN classifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print('Predicted:', knn.predict(iris_X_test))
print('Actual:   ', iris_y_test)


# Linear model--------------------------------------------------------
diabetes = datasets.load_diabetes()
diab_X_train = diabetes.data[:-20]
diab_X_test  = diabetes.data[-20:]
diab_y_train = diabetes.target[:-20]
diab_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diab_X_train, diab_y_train)

print('Coefs:', regr.coef_)

# MSE, r^2
print('MSE:', np.mean((regr.predict(diab_X_test) - diab_y_test) ** 2))
print('r^2:', regr.score(diab_X_test, diab_y_test))


# Shrinkage-----------------------------------------------------------
X = np.c_[0.5, 1].T
y = [0.5, 1]
test = np.c_[0, 2].T
regr = linear_model.LinearRegression()

plt.figure()
np.random.seed(191)

for _ in range(10):
    this_X  = 0.1 * np.random.normal(size = (2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s = 3)

plt.show()
    
regr = linear_model.Ridge(alpha = 0.1)

plt.figure()

for _ in range(10):
    this_X = 0.1 * np.random.normal(size = (2, 1)) + X
    regr.fit(this_X, y)
    plt.plot(test, regr.predict(test))
    plt.scatter(this_X, y, s = 3)

plt.show()

alphas = np.logspace(-4, -1, 6)
print([regr.set_params(alpha = alpha)\
       .fit(diab_X_train, diab_y_train)\
       .score(diab_X_test, diab_y_test)
       for alpha in alphas])

regr = linear_model.Lasso()
scores = [regr.set_params(alpha = alpha)\
          .fit(diab_X_train, diab_y_train)\
          .score(diab_X_test, diab_y_test)
          for alpha in alphas]
best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha

regr.fit(diab_X_train, diab_y_train)
print(regr.coef_)


# Classification------------------------------------------------------
logistic = linear_model.LogisticRegression(C = 1e5)
logistic.fit(iris_X_train, iris_y_train)

print('actual:   ', iris_y_test)
print('predicted:', logistic.predict(iris_X_test))


# SVM-----------------------------------------------------------------
svc = svm.SVC(kernel = 'linear')
svc.fit(iris_X_train, iris_y_train) # NOTE: should normalize data first
svc_poly = svm.SVC(kernel = 'poly', degree = 3)
svc_poly.fit(iris_X_train, iris_y_train)
svc_rbf = svm.SVC(kernel = 'rbf')
svc_rbf.fit(iris_X_train, iris_y_train)

print('actual:   ', iris_y_test)
print('predicted:')
print('linar svm:', svc.predict(iris_X_test))
print('poly3 svm:', svc_poly.predict(iris_X_test))
print('rbf svm:  ', svc_rbf.predict(iris_X_test))

