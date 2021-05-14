import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras.utils import to_categorical
from keras import models
from keras import layers
from tensorflow.python.keras.models import load_model

import sys

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def eval_ansamble(vecP):
    modelss = []
    ress = []
    n_models = 5
    for i in range(n_models):
        model:models.Sequential
        model = load_model('model' + str(i) + '.h5')
        modelss.append(model)

        ress.append(model.predict(np.array([vecP])))
    return ress

def getReview(path):
    s=""
    try:
        file = open(path)
        s = file.read()
    except Exception as e:
        print("failed to read file")
        return
    reviewStr = "".join(char for char in s if char.isalpha() or char.isspace() or char == "'").lower().strip().split()
    vecR = []

    index = imdb.get_word_index()

    for word in reviewStr:
        i = index.get(word)
        if i is not None and i < 500:
            vecR.append(i)

    vecP = vectorize(np.asarray([vecR]))

    return eval_ansamble(vecR)

if __name__ == "__main__":
    r = getReview(sys.argv[1])
    print([float(i) for i in r])
    print(sum(r)/len(r))




