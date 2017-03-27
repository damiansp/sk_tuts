from sklearn import svm

X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
print(clf.predict([[2., 2.]]))

# Get support vectors
print(clf.support_vectors_)

# Get indices of support vectors
print(clf.support_)

# Get no. support vectors for ea class
print(clf.n_support_)



# Multi-class Classification
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape = 'ovo')
clf.fit(X, Y)
dec = clf.decision_function([[1]])
print(dec.shape[1])

clf.decision_function_shape = 'ovr'
dec = clf.decision_function([[1]])
print(dec.shape[1])

