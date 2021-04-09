import numpy as np


def generator(data_size: int) -> np.ndarray:
    X = np.random.normal(loc=-5, scale=10, size=data_size)
    e = np.random.normal(loc=0, scale=.3, size=data_size)
    X_train = np.array([[
        -x_ ** 3 + e_,
        np.sin(3 * x_) + e_,
        np.exp(x_) + e_,
        x_ + 4 + e_,
        -x_ + np.sqrt(abs(x_)) + e_,
        x_ + e_
    ] for x_, e_ in zip(X, e)])
    y_train = np.array([[np.log(abs(x_)) + e_] for x_, e_ in zip(X, e)])
    return np.hstack((X_train, y_train))


def main():
    np.savetxt("data/train.csv", generator(1000), delimiter=',', fmt="%1.5f")
    np.savetxt("data/validation.csv", generator(100), delimiter=',', fmt="%1.5f")


if __name__ == '__main__':
    main()
