import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C = 1, kernel = 'linear')
print(svc.fit(X_digits[:-100], y_digits[:-100])\
      .score(X_digits[-100:], y_digits[-100:]))

X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()

for k in range(3):
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print scores


X = ['a', 'a', 'b', 'c', 'c', 'c']
k_fold = KFold(n_splits = 3)

for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' %(train_indices, test_indices))

kfold = KFold(n_splits = 3)
print([svc.fit(X_digits[train], y_digits[train])\
           .score(X_digits[test], y_digits[test])
       for train, test in k_fold.split(X_digits)])

cross_val_score(
    svc, X_digits, y_digits, cv = k_fold, scoring = 'precision_macro')



# Grid Search---------------------------------------------------------
Cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator = svc, param_grid = dict(C = Cs), n_jobs = -1)
clf.fit(X_digits[:1000], y_digits[:1000])

print(clf.best_score_)
print(clf.best_estimator_.C)
print(clf.score(X_digits[1000:], y_digits[1000:]))
print(cross_val_score(clf, X_digits, y_digits))



lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

lasso.fit(X_diabetes, y_diabetes)
print(lasso.alpha_)
