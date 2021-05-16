import re
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.datasets import imdb
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

MAX_REVIEW_LEN = 512
EMBEDDING_VECTOR_LEN = 32
MAX_WORDS = 5000
COLAB_PATH_PREFIX = "/content/drive/MyDrive/"


def get_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
    data = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)
    return train_test_split(data, targets, test_size=0.1)


def fit_model(f: Callable) -> Callable:
    def wrapper() -> (Sequential, float):
        model = f()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        results = model.fit(X_train, y_train, validation_split=0.2, epochs=2, batch_size=128)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        history = pd.DataFrame(results.history)
        plot(history[['loss', 'val_loss']], 'loss', 'Loss')
        plot(history[['accuracy', 'val_accuracy']], 'accuracy', 'Accuracy')
        return model, scores[1]

    return wrapper


def build_LSTM_model() -> Sequential:
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_VECTOR_LEN, input_length=MAX_REVIEW_LEN))
    model.add(LSTM(100))

    model.add(Dense(1, activation='sigmoid'))
    return model


def build_LSTM_CNN_model() -> Sequential:
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_VECTOR_LEN, input_length=MAX_REVIEW_LEN))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    return model


def plot(data: pd.DataFrame, label: str, title: str):
    axis = sns.lineplot(data=data, dashes=False)
    axis.set(ylabel=label, xlabel='epochs', title=title)
    axis.grid(True, linestyle="--")
    plt.show()


def load_custom_text(text: str) -> np.ndarray:
    text = text.lower()
    words = re.findall(r"\w+", text)
    index: dict = imdb.get_word_index()
    x = []
    for word in words:
        if word in index and index[word] < MAX_WORDS:
            x.append(index[word] + 3)
    x = sequence.pad_sequences([np.array(x)], maxlen=MAX_REVIEW_LEN)
    return x


def ensemble(models: List[Tuple[Sequential, float]], review: np.ndarray):
    scores = [m[1] for m in models]
    indices = np.argsort(scores)
    final_pred = 0
    for i in indices[::-1]:
        pred = models[i][0].predict(review)[0][0]
        print(f"Model #{i + 1}: {pred}")
        final_pred += (pred * (i + 1) / sum(range(len(models) + 1)))
    print(f"Final prediction: {final_pred}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_REVIEW_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_REVIEW_LEN)
    models = [
        fit_model(build_LSTM_model)(),
        fit_model(build_LSTM_CNN_model)(),
    ]
    with open(COLAB_PATH_PREFIX + "neg.txt", "r") as f_neg:
        ensemble(models, load_custom_text(f_neg.read()))

    with open(COLAB_PATH_PREFIX + "pos.txt", "r") as f_pos:
        ensemble(models, load_custom_text(f_pos.read()))
