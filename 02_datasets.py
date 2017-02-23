import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
print data.shape

digits = datasets.load_digits()
print digits.images.shape

plt.imshow(digits.images[0], cmap = plt.cm.gray_r)
plt.show()

data = digits.images.reshape((digits.images.shape[0], -1))


# Estimator objects---------------------------------------------------
#estimator = Estimator(param1 = 1, param2 = 2)
#print estimator.param1

#estimator.fit(data)
