import numpy as np


def gen_data(size=1000, mx=3, dx=10, me=0, de=0.3):
    size1 = size // 2
    size2 = size - size1

    x1 = np.reshape(np.random.normal(mx, dx, size1), (size1, 1))
    y1 = np.asarray([(x ** 3)/4 + np.random.normal(me, de, 1) for x in x1])

    x2 = np.reshape(np.random.normal(mx, dx, size1), (size2, 1))
    y2 = np.asarray([(x ** 3)/4 + np.random.normal(me, de, 1) for x in x2])

    return np.hstack((x1, y1)), np.hstack((x2, y2))


train, test = gen_data(10000)
np.savetxt("train.csv", train, delimiter=";")
np.savetxt("test.csv", test, delimiter=";")
