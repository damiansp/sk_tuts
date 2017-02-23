import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Ordinary Least Squares----------------------------------------------
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2]) # reg.fit(X, y)
print reg.coef_

# Example of Linear Regression----------------------------------------
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test  = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

print('Coefs:\n', regr.coef_)
print('Mean Squared Error: %.2f'
      %np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)))
print('R squared: %.2f' %regr.score(diabetes_X_test, diabetes_y_test))

# Plots
plt.scatter(diabetes_X_test, diabetes_y_test, color = 'k')
plt.plot(diabetes_X_test,
         regr.predict(diabetes_X_test),
         color = 'blue',
         linewidth = 3)
plt.xticks(())
plt.yticks(())
plt.show()



# Ridge Regression----------------------------------------------------
