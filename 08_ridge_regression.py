import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

reg = linear_model.Ridge(alpha = 0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])
print reg.intercept_
print reg.coef_


X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)
print X

n_alphas = 200
alphas = np.logspace(-10, 02, n_alphas)
clf = linear_model.Ridge(fit_intercept = False)
coefs = []

for a in alphas:
    clf.set_params(alpha = a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
#ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights (coefficients)')
plt.title('Ridge coefficients as a function of the regularization (alpha)')
plt.axis('tight')
plt.show()

# Using GCV to set regularization parameter---------------------------
reg = linear_model.RidgeCV(alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, 0.1, 1])

print reg.alpha_
