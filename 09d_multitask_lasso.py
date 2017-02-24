import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, MultiTaskLasso

rng = np.random.RandomState(85)

# Generate some 2D coefs with sine waves with rand freq and phase
n_samples, n_features, n_tasks = 100, 30, 40
n_relevant_features = 5
coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)

for k in range(n_relevant_features):
    coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))

X = rng.randn(n_samples, n_features)
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
coef_lasso = np.array([Lasso(alpha = 0.5).fit(X, y).coef_ for y in Y.T])
coef_multi_task_lasso = MultiTaskLasso(alpha = 1.).fit(X, Y).coef_

fig = plt.figure(figsize = (8, 5))
plt.subplot(1, 2, 1) # 1 x 2; 1st
plt.spy(coef_lasso)
plt.xlabel('Feature')
plt.ylabel('Time or Task')
plt.text(10, 5, 'Lasso')

plt.subplot(1, 2, 2)
plt.spy(coef_multi_task_lasso)
plt.xlabel('Feature')
plt.ylabel('Time or Task')
plt.text(10, 5, 'Multitask Lasso')
fig.suptitle('Coefficient non-zero location')


plt.figure()
plt.plot(coef[:, 0], color = 'seagreen', linewidth = 2, label = 'Ground truth')
plt.plot(
    coef_lasso[:, 0], color = 'cornflowerblue', linewidth = 2, label = 'Lasso')
plt.plot(coef_multi_task_lasso[:, 0],
         color = 'gold',
         linewidth = 2,
         label = 'Multitask Lasso')
plt.legend(loc = 'best')
plt.axis('tight')
plt.ylim([-1.1, 1.1])
plt.show()

