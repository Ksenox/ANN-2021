import numpy as np


def gen(n):
    x_data = np.random.normal(0, 10, n)
    e_data = np.random.normal(0, 0.3, n)

    fn = [
        lambda x, e: np.sin(x / 2) + e,
        lambda x, e: np.cos(x * 2) + e,
        lambda x, e: x - 3 + e,
        lambda x, e: -x + e,
        lambda x, e: np.abs(x) + e,
        lambda x, e: np.power(x, 3) / 4 + e,
    ]

    f = np.vectorize(lambda i, j: fn[i](x_data[j], e_data[j]))

    data = np.reshape(np.array([
        [f(range(6), j)]
        for j in range(n)]
    ), (n, 6))

    label = np.transpose(np.array([
        np.power(x_data, 2) + e_data
    ]))

    return np.append(data, label, axis=1)


np.savetxt('train.csv', gen(1000), delimiter=';', fmt='%1.5f')
np.savetxt('test.csv', gen(300), delimiter=';', fmt='%1.5f')



