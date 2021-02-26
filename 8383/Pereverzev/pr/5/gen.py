import numpy as np
from math import sin

sizeStud = 700
sizeValid = 700
x = [-5, 10]
e = [0, 0.03]
dx = x[1]-x[0]


def v2_3(X):
    return[(3*x[0]) + np.random.rand(1) * e[1] for x in X]


def gen(size):
    localX = [(np.random.rand(1) * (x[1]-x[0]))+x[0]
              for idx in range(size)]
    localVal = v2_3(localX)
    return np.hstack((np.asarray(localX), np.asarray(localVal)))


np.savetxt("stud.csv", gen(sizeStud), delimiter=",")
np.savetxt("valid.csv", gen(sizeValid), delimiter=",")
