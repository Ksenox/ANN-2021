import numpy as np
import math

def calcValue(X, e, size):
    data = np.zeros([size, 7])
    for i in range(size):
        f1 = math.pow(X[i], 2) + e[i]
        f2 = math.sin(X[i] / 2) + e[i]
        f3 = math.cos(2 * X[i]) + e[i]
        f4 = X[i] - 3 + e[i]
        f5 = - X[i] + e[i]
        f6 = math.fabs(X[i]) + e[i]
        f7 = (math.pow(X[i], 3)) / 4 + e[i]
        data[i] = np.array([f1, f2, f3, f4, f5, f6, f7])

    return data

train_size = 2000
test_size = 500

#генерация X, e:
mu_X = 3
sigma_X = 10
X_train = np.random.normal(mu_X, sigma_X, train_size)
X_test = np.random.normal(mu_X, sigma_X, test_size)

mu_e = 0
sigma_e = 0.3
e_train = np.random.normal(mu_e, sigma_e, train_size)
e_test = np.random.normal(mu_e, sigma_e, test_size)

data_train = calcValue(X_train, e_train, train_size)
np.savetxt(fname="data_train.csv", X=data_train, delimiter=', ')
data_test = calcValue(X_test, e_test, test_size)
np.savetxt(fname="data_test.csv", X=data_test, delimiter=', ')