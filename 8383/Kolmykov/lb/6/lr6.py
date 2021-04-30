import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
import re


def interface(model, vec_len):
    index = imdb.get_word_index()
    while (True):
        print('Input text (or "stop"):')
        text = input()
        text = clear_text(text)
        if text == 'stop':
            break

        ind_arr = get_indexes_from_text(text, index, vec_len)
        vec = vectorize(np.asarray([ind_arr]))
        res = model.predict(vec)
        if res >= 0.5:
            print('Positive')
        else:
            print('Negative')


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def clear_text(text):
    text = re.sub(r'([^A-z ]|[\[\]])', '', text).lower()
    text = text.strip()
    text = re.sub(r'  +', ' ', text)
    return text


def get_indexes_from_text(text, index, vec_len):
    text_arr = text.split(" ")
    ind_arr = []
    for word in text_arr:
        num = index.get(word)
        if num is not None and num < vec_len:
            ind_arr.append(num + 3)
    return ind_arr

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

vec_len = 10_000

data = vectorize(data, vec_len)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(vec_len,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
results = model.fit(
    train_x, train_y,
    epochs=2,
    batch_size=500,
    validation_data=(test_x, test_y),
    verbose=False
)

interface(model, vec_len)
