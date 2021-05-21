import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.datasets import imdb
from tensorflow.python.keras.models import load_model

from lb6.main import vectorize

model = load_model("models/2.h5")


def input_():
    filename = input("Input filename of review: ")
    with open(filename, "r") as f:
        predict(f.read())


def prepare(content, dim):
    words = [word.strip(":;?!.,'\"-_").lower() for word in content.strip().split()]
    index = imdb.get_word_index()
    data = []
    for word in words:
        if word in index and index[word] < dim:
            data.append(index[word])
    data = vectorize([np.array(data)], dim)
    return data


def prediction_message(prediction):
    if prediction > 0.7:
        print("This review is a good one")
    elif prediction < 0.3:
        print("This review is a bad one")
    else:
        print("Cannot determine a class of the review")


def predict(text):
    x = prepare(text, 10000)
    prediction = model.predict(x)
    print(f"Prediction for this review is {prediction}")
    prediction_message(prediction)
