import numpy as np
#v2_5

def f1(x, e):
    return pow(-x, 3)+e


def f2(x, e):
    return np.log(abs(x))+e


def f3(x, e):
    return np.sin(3*x)+e


def f4(x, e):
    return np.exp(x)+e


def f5(x, e):
    return x+4+e


def f6(x, e):
    return -x+np.sqrt(abs(x))+e


def f7(x, e):
    return x+e

m_x = 0
x_e = 0
s_x = 10
s_e = 0.3
train_size = 1000
test_size = 300
X = np.random.normal(m_x, s_x, train_size)
E = np.random.normal(m_x, s_x, train_size)
train_data = np.array([[f1(X[i], E[i]), f2(X[i], E[i]), f3(X[i], E[i]), f4(X[i], E[i]), f6(X[i], E[i]), f7(X[i], E[i])] for i in range(train_size)])
train_labels = np.array([f5(X[i], E[i]) for i in range(train_size)])
train_labels = np.reshape(train_labels, (train_size, 1))
train_data = np.hstack((train_data, train_labels))

X = np.random.normal(m_x, s_x, test_size)
E = np.random.normal(m_x, s_x, test_size)
test_data = np.array([[f1(X[i], E[i]), f2(X[i], E[i]), f3(X[i], E[i]), f4(X[i], E[i]), f6(X[i], E[i]), f7(X[i], E[i])] for i in range(test_size)])
test_labels = np.array([f5(X[i], E[i]) for i in range(test_size)])
test_labels = np.reshape(test_labels, (test_size, 1))
test_data = np.hstack((test_data, test_labels))

np.savetxt("train_data.csv", train_data, delimiter=",")
np.savetxt("test_data.csv", test_data, delimiter=",")
