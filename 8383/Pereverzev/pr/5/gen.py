import numpy as np
import pandas as pd


def genData(size_of_dataset, inp_size):
    dataset = np.zeros((size_of_dataset, inp_size))
    dataset_y = np.zeros(size_of_dataset)
    for i in range(size_of_dataset):
        X = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        dataset[i, :] = (
            np.round(e - X**3),
            np.round(e + np.log(np.abs(X))),
            np.round(e + np.exp(X)),
            np.round(e + 4 + X),
            np.round(e + np.sqrt(np.abs(X)) - X),
            np.round(e + X),
        )
        dataset_y[i] = np.round(e + (3*X))
    return np.round(np.array(dataset), decimals=3), np.array(dataset_y)


def genFullData(inp_size):
    x_train, y_train = genData(600, inp_size)
    x_test, y_test = genData(200, inp_size)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train -= y_mean
    y_train /= y_std
    y_test -= y_mean
    y_test /= y_std

    pd.DataFrame(np.round(x_test, 3)).to_csv("./dataset/x_test.csv")
    pd.DataFrame(np.round(y_test, 3)).to_csv("./dataset/y_test.csv")
    pd.DataFrame(np.round(x_train, 3)).to_csv("./dataset/x_train.csv")
    pd.DataFrame(np.round(y_train, 3)).to_csv("./dataset/y_train.csv")

    return x_train, y_train, x_test, y_test
