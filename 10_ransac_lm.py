import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets

n_samples  = 1000
n_outliers = 50
X, y, coef = datasets.make_regression(n_samples = n_samples,
                                      n_features = 1,
                                      n_informative = 1,
                                      noise = 10,
                                      coef = True,
                                      random_state = 11235)
np.random.seed(12358)
X[:n_outliers] =  3 + 0.5 * np.random.normal(size = (n_outliers, 1))
y[:n_outliers] = -3 + 10  * np.random.normal(size = n_outliers)

# Fit line to all data
mod_lm = linear_model.LinearRegression()
mod_lm.fit(X, y)

# Robust fit with RANSAC (RANdom SAmple Consensus)
mod_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
mod_ransac.fit(X, y)
inlier_mask = mod_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict est
line_X = np.arange(-5, 5)
line_y        = mod_lm.predict(line_X[:, np.newaxis])
line_y_ransac = mod_ransac.predict(line_X[:, np.newaxis])

# Compare coefs
print('Estimated coeffs (true, normal, RANSAC):')
print(coef, mod_lm.coef_, mod_ransac.estimator_.coef_)

# Plot
lw = 2
plt.scatter(X[inlier_mask],
            y[inlier_mask],
            color = 'yellowgreen',
            marker = '.',
            label = 'Inliers')
plt.scatter(X[outlier_mask],
            y[outlier_mask],
            color = 'gold',
            marker = '.',
            label = 'Outliers')
plt.plot(line_X,
         line_y,
         color = 'navy',
         linestyle = '-',
         linewidth = lw,
         label = 'Linear Regressor')
plt.plot(line_X,
         line_y_ransac,
         color = 'cornflowerblue',
         linestyle = '-',
         linewidth = lw,
         label = 'RANSAC Regressor')
plt.legend(loc = 'lower right')
plt.show()
