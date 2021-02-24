import numpy as np


def gen_data(size=1000, x_down=3, x_up=10, e_up=0.3):
    size1 = size // 2
    size2 = size - size1

    x1 = np.asarray([np.random.rand(1) * (x_up - x_down) + x_down for i in range(size1)])
    y1 = np.asarray([(x ** 3)/4 + np.random.rand(1) * e_up for x in x1])

    x2 = np.asarray([np.random.rand(1) * (x_up - x_down) + x_down for i in range(size2)])
    y2 = np.asarray([(x ** 3) / 4 + np.random.rand(1) * e_up for x in x2])

    return np.hstack((x1, y1)), np.hstack((x2, y2))


train, test = gen_data()
np.savetxt("train.csv", train, delimiter=";")
np.savetxt("test.csv", test, delimiter=";")
