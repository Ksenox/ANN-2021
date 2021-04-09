import numpy as np


def f1(x, e):
    return x**2 + e


def f2(x, e):
    return np.sin(x/2) + e


def f3(x, e):
    return np.cos(2*x) + e


def f4(x, e):
    return x - 3 + e


def f5(x, e):
    return -x + e


def f6(x, e):
    return abs(x) + e


def f7(x, e):
    return (x**3)/4 + e


funcs = [f1, f3, f4, f5, f6, f7, f2]

train_size = 2500
test_size = 500

X_from = 3
X_to = 10
X_data_train = np.random.normal(X_from, X_to, train_size)
X_data_test = np.random.normal(X_from, X_to, test_size)

e_from = 0
e_to = 0.3
e_data_train = np.random.normal(e_from, e_to, train_size)
e_data_test = np.random.normal(e_from, e_to, test_size)

train_data = np.array([[funcs[j](X_data_train[i], e_data_train[i]) for j in range(7)] for i in range(train_size)])
test_data = np.array([[funcs[j](X_data_test[i], e_data_test[i]) for j in range(7)] for i in range(test_size)])

np.savetxt('train_data.csv', train_data, delimiter=' ', fmt='%1.5f')
np.savetxt('test_data.csv', test_data, delimiter=' ', fmt='%1.5f')