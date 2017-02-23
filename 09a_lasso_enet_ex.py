import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score

# Lasso & Elastic Net Example
n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 5 * np.random.randn(n_features)
inds = np.arange(n_features)

np.random.shuffle(inds)
coef[inds[10:]] = 0 # make coefs sparse
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# Train/Test
n_samples = X.shape[0]
X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
X_test, y_test   = X[n_samples / 2:], y[n_samples / 2:]

alpha = 0.1
lasso = Lasso(alpha = alpha)
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print(lasso)
print('r^2 on test data: %.3f' %r2_score_lasso)


# Elastic Net
enet = ElasticNet(alpha = alpha, l1_ratio = 0.7)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

print(enet)
print('r^2 on test data: %.3f' % r2_score_enet)

plt.plot(enet.coef_,  color = 'green', linewidth = 2, label = 'Elastic net')
plt.plot(lasso.coef_, color = 'red',   linewidth = 2, label = 'Lasso')
plt.plot(coef,        color = 'black', label = 'Actual')
plt.legend(loc = 'best')
plt.title('Lasso R^2: %.3f, Elastic Net R^2: %.3f'
          %(r2_score_lasso, r2_score_enet))
plt.show()
