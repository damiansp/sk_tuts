from __future__ import print_function
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from time import time
#import logging
import matplotlib.pyplot as plt
import numpy as np

logistic = linear_model.LogisticRegression()
pca = PCA()
pipe = Pipeline(steps = [('pca', pca), ('logistic', logistic)])
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# Plot the PCA spectrum
pca.fit(X_digits)

plt.figure(1, figsize = (4, 3))
plt.clf()
plt.axes([0.2, 0.2, 0.7, 0.7])
plt.plot(pca.explained_variance_, linewidth = 2)
plt.axis('tight')
plt.xlabel('n components')
plt.ylabel('explained variance')

# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)

# Params of pipeline can be set using '__' separated param names
estimator = GridSearchCV(
    pipe, dict(pca__n_components = n_components, logistic__C = Cs))

estimator.fit(X_digits, y_digits)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle = ':',
            label = 'n components chosen')
plt.legend(prop = dict(size = 12))
plt.show()



# Facial Recognition with Eigenfaces----------------------------------
#print(__doc__)

lfw_people = datasets.fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = targget_names.shape[0]

print('Total data set size: ')
print('n_samples: %d'   %n_samples)
print('n_features: %d' %n_features)
print('n_classes: %d'  %n_classes)

# Split int test/train with k-folds
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 42)

# Compute PCA (eigenface) on the data set
n_components = 150
print('Extracting the top %d eigenfaces from %d faces'
      %(n_components, X_train.shape[0]))
t0 = time()
pca = PCA(
    n_components = n_components, svd_resolver = 'randomized', whiten = True)\
    .fit(X_train)
print('done in %0.3fs' %(time() - t0))
eigenfaces = pca.components_.reshape((n_components, h, w))
print('Projecting the input data on the eigenface orthonormal basis')
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)
print('done in %0.3fs' %(time() - t0))

# Train SVM
print('Fitting the classifier to the training set')
t0 = time()
param_grid = { 'C': [1e3, 5e3, 1e4, 5e4, 1e5],
               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], }
clf = GridSearchCV(SVC(kernel = 'rbf', class_weight = 'balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print('done in %0.3fs' %(time() - t0))
print('Best estimator found by grid search:\n', clf.best_estimator_)

# Evaluate model
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print('done in %0.3fs' %(time() - t0))
print(classification_report(y_test, y_pred, target_names = target_names))
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))

# Vis
def plot_gallery(images, titles, h, w, n_row = 3, n_col = 4):
    '''Helper function to plot a gallery of portraits'''
    plt.figure(figsize = (1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(
        bottom = 0, left = 0.01, right = 0.99, top = 0.9, hspace = 0.35)

    for i in range(n_row * ncol):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray)
        plt.title(titles[i], size = 12)
        plt.xticks(())
        plt.yticks(())

# Plot result of pred on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' %(pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# Plot the most significant eigenfaces
eigenface_titles = ['Eigenface %d' %i for i in range(eigenfaces.shape[0])]

plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
