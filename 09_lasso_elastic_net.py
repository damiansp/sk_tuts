import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score

reg = Lasso(alpha = 0.1)

reg.fit([[0, 0], [1, 1]], [0, 1])
print reg.predict([[1, 1]])

