import numpy as np
import math


def gen_data(size=1000, mx=3, dx=10, me=0, de=0.3):
    size1 = size // 2
    size2 = size - size1

    train_data = gen(size1, mx, dx, me, de)
    test_data = gen(size2, mx, dx, me, de)

    return train_data, test_data


def gen(size, mx=3, dx=10, me=0, de=0.3):
    all_x = np.reshape(np.random.normal(mx, dx, size), (size, 1))
    print("X:")
    print(all_x)
    e = np.reshape(np.random.normal(me, de, size), (size, 1))
    inp = np.asarray([[f1(all_x[i], e[i]), f2(all_x[i], e[i]), f3(all_x[i], e[i]),
                       f4(all_x[i], e[i]), f5(all_x[i], e[i]), f6(all_x[i], e[i])] for i in range(size)])
    inp = np.reshape(inp, (size, 6))
    out = np.asarray([f7(x, np.random.normal(me, de, 1)) for x in all_x])
    return np.hstack((inp, out))


def f1(x, e):
    return x**2 + e


def f2(x, e):
    return math.sin(x/2) + e


def f3(x, e):
    return math.cos(2 * x) + e


def f4(x, e):
    return x - 3 + e


def f5(x, e):
    return -x + e


def f6(x, e):
    return math.fabs(x) + e


def f7(x, e):
    return (x**3)/4 + e


train, test = gen_data(2000)
np.savetxt("train.csv", train, delimiter=";")
np.savetxt("test.csv", test, delimiter=";")
