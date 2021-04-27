import numpy as np
from tensorflow.keras.models import Sequential
from keras import layers
import tensorflow as tf
from keras.datasets import imdb
import re
from keras.models import model_from_json


def vectorize(sequences, dimension=20000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    optim = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=True, clipnorm=1.,
                                   clipvalue=0.5)
    loaded_model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
    return loaded_model


def user_input():
    print("Type file name:\n")
    filename = input()
    f = open(filename+".txt", "r")
    text = f.read()
    prediction(text_prepare(text), load_model())
    return 0


def prediction(coded, model):
    res = model.predict(coded)
    print(res)
    if res >= 0.5:
        print('good')
    else:
        print('bad')


def text_prepare(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = text.split(' ')
    index = imdb.get_word_index()
    coded = []

    for word in text:
        ind = index.get(word)
        if ind is not None and ind < 20000:
            coded.append(ind + 3)
    return vectorize(np.asarray([coded]))

def create_model():
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data)
    targets = np.array(targets).astype("float32")

    test_x = data[:5000]
    test_y = targets[:5000]
    train_x = data[5000:]
    train_y = targets[5000:]

    model = Sequential()
    model.add(layers.Dense(50, activation="sigmoid", input_shape=(20000,)))
    model.add(layers.Dense(50, activation="sigmoid"))
    model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
    model.add(layers.Dense(75, activation="sigmoid"))
    model.add(layers.Dense(1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=True, clipnorm=1.,
                                   clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    hist = model.fit(train_x, train_y, epochs=2, batch_size=1000, validation_data=(test_x, test_y))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")


user_input()
