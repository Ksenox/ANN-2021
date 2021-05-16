import os

import sklearn

from pr8.callbacks import FeatureMapForEpochCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical

from pr8.src.var7 import gen_data


def build_model(layers):
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def show_image(image, label):
    plt.imshow(image.reshape(image.shape[0], image.shape[0]), cmap=plt.cm.binary)
    plt.show()
    print(label)


def plot(epochs, train, validation, metrics):
    plt.plot(epochs, train, 'b', label=f'Training {metrics}')
    plt.plot(epochs, validation, 'r', label=f'Validation {metrics}')
    plt.title(f'Training and validation {metrics}')
    plt.xlabel('Epochs')
    plt.ylabel(metrics.capitalize())
    plt.grid(True)
    plt.legend()


def plot_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.subplot(211)
    plot(epochs, loss, val_loss, "loss")
    plt.subplot(212)
    plot(epochs, acc, val_acc, "accuracy")
    plt.show()


def normalize(fn):
    def wrapper():
        x, y = fn()
        x /= np.max(x)
        return x, y

    return wrapper


def one_hot_y(fn):
    def wrapper():
        x, y = fn()
        return x, to_categorical(y)

    return wrapper


def encode_labels(fn):
    def wrapper():
        x, y = fn()
        return x, LabelEncoder().fit_transform(y)

    return wrapper


def shuffle(fn):
    def wrapper():
        x, y = fn()
        x, y = sklearn.utils.shuffle(x, y)
        return x, y

    return wrapper


def train_test(fn):
    def wrapper():
        global train_size, image_side
        x, y = fn()
        x = x.reshape(-1, image_side, image_side, 1)
        train_len = int(x.shape[0] * train_size)
        return x[:train_len, :], y[:train_len, :], x[train_len:, :], y[train_len:, :]

    return wrapper


image_side = 50
n_samples = 2000
train_size = 0.9


@train_test
@normalize
@one_hot_y
@shuffle
@encode_labels
def prepare_data():
    global n_samples, image_side
    return gen_data(n_samples, image_side)


train_x, train_y, test_x, test_y = prepare_data()


# X, y = gen_data()
# X, y = sklearn.utils.shuffle(X, y)
# for i in range(222, 260):
#     plt.imshow(X[i], cmap=plt.cm.binary)
#     plt.savefig(f"samples/{i}_{y[i]}.png")


layers = [
    Input(shape=(image_side, image_side, 1)),
    Convolution2D(filters=32, kernel_size=(7, 7), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(5, 5), padding="same"),
    Convolution2D(filters=64, kernel_size=(7, 7), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(5, 5), padding="same"),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.25),
    Dense(256, activation="relu"),
    Dropout(0.15),
    Dense(3, activation="softmax")
]

model = build_model(layers)
history = model.fit(train_x, train_y, batch_size=20, epochs=10, validation_split=0.1, callbacks=[FeatureMapForEpochCallback([1])])
model.evaluate(test_x, test_y)
plot_history(history.history)
