import numpy as np


def task1(a, b):
    if len(a) != len(b): raise IndexError()  # проверка на соответсвие количества матриц и векторов

    total = np.zeros([b.shape[1], 1], dtype=int)  # создание нулевого вектора

    for i in range(0, len(a), 1):  # идем по всем векторам и матрицам из множеств
        total = total + a[i].dot(b[i])  # умножаем матрицу a[i] на вектор b[i]
        # и прибавляем полученные вктор к пполученным ранее

    if total.shape[1] != 1: raise ValueError()  # проверка, что исходный вектор имеет размерность (n, 1)

    return total


try:

    buffer = np.genfromtxt('testFile.txt', dtype='int', delimiter=' ')

    a = np.reshape(buffer[:len(buffer) - 3], [3, 4, 4])
    b = np.reshape(buffer[len(buffer) - 3:], [3, 4, 1])

    answer = task1(a, b)

    np.savetxt('fileOut.txt', answer, fmt='%d')

except IndexError:
    print("1 Размерность  или количество векторов или матриц не верная")
except AttributeError:
    print("2 Размерность векторов или матриц не верная")
except ValueError:
    print("3 Размерность векторов или матриц не верная")
except TypeError:
    print("4 Размерность векторов или матриц не верная")





