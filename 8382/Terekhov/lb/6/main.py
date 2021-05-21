import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
from sklearn.model_selection import train_test_split

COLAB_PATH_PREFIX = "/content/drive/MyDrive/"
SKIP_WORDS = 50
NUM_WORDS = 10000


def plot(data: pd.DataFrame, label: str, title: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=title)
    axis.grid(True, linestyle="--")
    plt.show()


def decode_report(report: List[int]) -> str:
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in report])
    return decoded


def vectorize(sequences: List[List[int]], dimension: int = NUM_WORDS) -> np.ndarray:
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, np.array(sequence) - SKIP_WORDS] = 1
    return results


def build_model() -> Sequential:
    model = Sequential()
    model.add(layers.Dense(100, activation="relu", input_shape=(NUM_WORDS,)))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def save_sorted_index():
    index = imdb.get_word_index()
    sorted_index = dict(sorted(index.items(), key=lambda x: x[1]))
    with open("fout.csv", "w") as f:
        f.writelines([f"{i}, {w}\n" for w, i in sorted_index.items()])


def load_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


def load_custom_text(text: str, dimension: int = NUM_WORDS) -> np.ndarray:
    text = text.lower()
    words = re.findall(r"\w+", text)
    index: dict = imdb.get_word_index()
    x = []
    for word in words:
        if word in index and index[word] < dimension:
            x.append(index[word])
    x = vectorize([np.array(x)])
    return x


if __name__ == '__main__':
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(
        num_words=NUM_WORDS + SKIP_WORDS, skip_top=SKIP_WORDS)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data)
    print(data.shape)
    targets = np.array(targets).astype("float32")
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    results = model.fit(x_train, y_train, epochs=5, batch_size=4096, validation_data=(x_test, y_test))
    history = pd.DataFrame(results.history)
    plot(history[['loss', 'val_loss']], 'loss', 'Loss')
    plot(history[['accuracy', 'val_accuracy']], 'accuracy', 'Accuracy')
    files = [COLAB_PATH_PREFIX + "neg.txt", COLAB_PATH_PREFIX + "pos.txt"]
    for file in files:
        pred = model.predict(load_custom_text(load_file(file)))
        print(pred, "Positive" if pred > 0.5 else "Negative")