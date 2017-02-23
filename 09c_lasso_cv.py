import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
rng = np.random.RandomState(7)
X = np.c_[X, rng.randn(X.shape[0], 14)] # add some bad features

# Normalize data as done in Lars for direct comparison of results
X /= np.sqrt(np.sum(X ** 2, axis = 0))


# LassoLarsIC: least angle regression wih A/BIC
model_aic = LassoLarsIC(criterion = 'aic')
model_aic.fit(X, y)
alpha_aic = model_aic.alpha_

t1 = time.time()
model_bic = LassoLarsIC(criterion = 'bic')
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic = model_bic.alpha_

def plot_ic(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_

    plt.plot(-np.log10(alphas_),
             criterion_,
             color = color,
             linewidth = 2,
             label = '%s criterion' %name)

plt.figure()
plot_ic(model_aic, 'AIC', 'b')
plot_ic(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information criteria for model selection (training time %.3fs)'
          %t_bic)
plt.show()


# Lasso CV with Coordinate Descent
t1 = time.time()
model = LassoCV(cv = 20).fit(X, y)
t_lasso_cv = time.time() - t1
m_log_alphas = -np.log10(model.alphas_)

ymin, ymax = 2300, 3800
plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas,
         model.mse_path_.mean(axis = -1),
         'k',
         label = 'Mean across folds',
         linewidth = 2)
plt.axvline(-np.log10(model.alpha_),
            linestyle = '--',
            color = 'k',
            label = 'alpha: CV estimate')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('MSE')
plt.title('MSE on each fold: coordinate descent (train time: %.2fs)'
          %t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)
plt.show()
