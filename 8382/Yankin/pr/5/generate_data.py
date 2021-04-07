import numpy as np


def f1(x, e):
    return -x**3 + e


def f2(x, e):
    return np.log(abs(x)) + e


def f3(x, e):
    return np.sin(3*x) + e


def f4(x, e):
    return np.exp(x) + e


def f5(x, e):
    return x + 4 + e


def f6(x, e):
    return -x + np.sqrt(abs(x)) + e


def f7(x, e):
    return x + e


def generate(size):
    x = np.random.uniform(-5, 10, size)
    e = np.random.uniform(0, 0.03, size)

    data = np.asarray([
        [f1(x[i], e[i]), f2(x[i], e[i]), f4(x[i], e[i]), f5(x[i], e[i]), f6(x[i], e[i]), f7(x[i], e[i])]
        for i in range(size)])
    labels = np.asarray([f3(x[i], e[i]) for i in range(size)])
    data = np.hstack((data, np.reshape(labels, (size, 1))))

    return data


np.savetxt('train.csv', X=generate(1000), delimiter=';')
np.savetxt('test.csv', X=generate(200), delimiter=';')
