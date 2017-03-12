import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy import linalg
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)

# Color map
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    { 'red':   [(0, 1,   1),   (1, 0.7, 0.7)],
      'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
      'blue':  [(0, 0.7, 0.7), (1, 1,   1)] })
plt.cm.register_cmap(cmap = cmap)

# Generate data
def dataset_fixed_cov():
    '''Generate 2 Gaussian samples with same covariance matrix'''
    np.random.seed(110376)

    n, dim = 300, 2
    C = np.array([[0., -0.23], [0.82, 0.23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def dataset_cov():
    '''Generate 2 Gaussian samples with different covariance matrices'''
    np.random.seed(110376)

    n, dim = 300, 2
    C = np.array([[0., -1.], [2.5, 0.7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

# Plot
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)

    if fig_index == 1:
        plt.title('LDA')
        plt.ylabel('Data with fixed covariance')
    elif fig_index == 2:
        plt.title('QDA')
    elif fig_index == 3:
        plt.ylabel('Data with different covariances')

    tp = (y == y_pred) # True positves
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    alpha = 0.2

    # Class 0: points
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha = alpha, color = 'red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha = alpha, color = '#990000')

    # Class 1: points
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha = alpha, color = 'blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha = alpha, color = '#000099')
    
    # Class 0/1: areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    plt.contour(xx, yy, Z, [0.5], linewidths = 1., colors = 'k')

    # Means
    plt.plot(lda.means_[0][0],
             lda.means_[0][1],
             'o',
             color = 'black',
             markersize = 4)
    plt.plot(lda.means_[1][0],
             lda.means_[1][1],
             'o',
             color = 'black',
             markersize = 4)
    return splot

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi # to degrees

    # Filled Gaussian at 2sd
    ell = mpl.patches.Ellipse(mean,
                              2 * v[0] ** 0.5,
                              2 * v[1] ** 0.5,
                              180 + angle,
                              facecolor = color,
                              edgecolor = 'yellow',
                              linewidth = 1,
                              zorder = 2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())

def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')

def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')

for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # LDA
    lda = LinearDiscriminantAnalysis(solver = 'svd', store_covariance = True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index = 2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # QDA
    qda = QuadraticDiscriminantAnalysis(store_covariances = True)
    y_pred = qda.fit(X, y).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index = 2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')

plt.suptitle('Linear vs Quadratic Discriminant Analysis')
plt.show()
