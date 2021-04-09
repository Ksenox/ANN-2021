import numpy as np


def f1(x, e):
    return x**2 + x + e


def f2(x, e):
    return np.abs(x) + e


def f3(x, e):
    return np.sin(x - np.pi / 4) + e


def f4(x, e):
    return np.log(abs(x)) + e


def f5(x, e):
    return -np.power(x, 3) + e


def f6(x, e):
    return -x/4 + e


def f7(x, e):
    return -x + e


def gen_data(num_data=1000):
    X = np.random.normal(loc=0, scale=10, size=num_data)
    E = np.random.normal(loc=0, scale=0.3, size=num_data)
    data_train = np.array([[f2(X[i], E[i]), f3(X[i], E[i]), f4(X[i], E[i]),
    f5(X[i], E[i]), f6(X[i], E[i]), f7(X[i], E[i])] for i in range(num_data)])
    data_labels = np.reshape(np.array([f1(X[i], E[i]) for i in range(num_data)]), (num_data, 1))
    data = np.hstack((data_train, data_labels))
    return data


train = gen_data(1000)
validation = gen_data(200)

np.savetxt("train.csv", train, delimiter=";")
np.savetxt("validation.csv", validation, delimiter=";")