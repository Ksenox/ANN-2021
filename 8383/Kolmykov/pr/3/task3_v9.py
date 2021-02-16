import numpy as np


def f(matrix):
    result = np.zeros(matrix.shape[0], dtype='int')
    for i in range(len(result)):
        result[i] = getResForLine(matrix[i])
    return result


def getResForLine(line):
    dictionary = {}
    for num in line:
        if dictionary.get(num) is not None:
            dictionary[num] = dictionary[num] + 1
        else:
            dictionary[num] = 1
    return getMaxInDict(dictionary)


def getMaxInDict(dictionary):
    values = dictionary.values()
    maxVal = list(values)[0]
    for val in values:
        if val > maxVal:
            maxVal = val
    return getKeyByValue(dictionary, maxVal)


def getKeyByValue(dictionary, val):
    for pair in dictionary.items():
        if pair[1] == val:
            return pair[0]


# print(f(np.array([[1, 2, 3], [4, 5, 5], [6, 6, 6], [7, 7, 8]])))
matrix = np.genfromtxt('task3.csv', dtype='int', delimiter=';')
res = f(matrix)
print(res)
file = open('res.txt', 'w')
file.write(str(res))
