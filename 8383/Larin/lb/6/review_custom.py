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

def getReview(path):
    s=""
    try:
        file = open(path)
        s = file.read()

    except Exception as e:
        print("failed to rean file")
        return

    reviewStr = "".join(char for char in s if char.isalpha() or char.isspace() or char == "'").lower().strip().split()
    vecR = []

    index = imdb.get_word_index()

    for word in reviewStr:
        i = index.get(word)
        if i is not None and i < 10000:
            vecR.append(i)

    vecP = vectorize(np.asarray([vecR]))

    model = load_model("model.h5")
    p = model.predict(vecP)
    return p

if __name__ == "__main__":
    print(getReview(sys.argv[1]))




