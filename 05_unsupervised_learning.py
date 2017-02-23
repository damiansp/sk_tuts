import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import misc
from sklearn import cluster, datasets, decomposition
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.utils.fixes import sp_version
from sklearn.utils.testing import SkipTest


# Clustering----------------------------------------------------------
# K-means
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
k_means = cluster.KMeans(n_clusters = 3)

k_means.fit(X_iris)
print(y_iris[::10])
print(k_means.labels_[::10])

# Example: vector quantization
face = misc.face(gray = True)
X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters = 5, n_init = 10)

k_means.fit(X)

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
face_compressed.shape = face.shape

# Hierarchical Agglomerative Clustering: Ward
face = sp.misc.imresize(face, 0.10) / 255

digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)
agglom = cluster.FeatureAgglomeration(connectivity = connectivity,
                                      n_clusters = 32)
agglom.fit(X)
X_reduced = agglom.transform(X)
X_approx = agglom.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)



# Decompositions------------------------------------------------------
# Create a signal with only 2 useful dimensions
# PCA
x1 = np.random.normal(size = 100)
x2 = np.random.normal(size = 100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]
pca = decomposition.PCA()

pca.fit(X)
print(pca.explained_variance_)

pca.n_components = 2
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)


# Independent Component Analysis (ICA)
# Generate sample data
time = np.linspace(0, 10, 2000)
s1 = np.sin(2 * time)          # sinewave
s2 = np.sign(np.sin(3 * time)) # square wave
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size = S.shape) # add noise
S /= S.std(axis = 0)                        # standardize
# Mix data
A = np.array([[1, 1], [0.5, 2]]) # Mix matrix
X = np.dot(S, A.T)

# Compute ICA
ica = decomposition.FastICA()
S_ = ica.fit_transform(X)
A_ = ica.mixing_.T
print(np.allclose(X, np.dot(S_, A_) + ica.mean_))
