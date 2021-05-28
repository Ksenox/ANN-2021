import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Model, Sequential
from keras.utils import np_utils
from tensorflow.python.keras.callbacks import TensorBoard

FILENAME = "wonderland.txt"
FILENAME_PATTERN = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"


class TextGeneratorCallback(Callback):
    def __init__(self, int_to_char: Dict[int, str], data_x: np.array, n_vocab) -> None:
        super().__init__()
        self._int_to_char = int_to_char
        self._data_x = data_x
        self._n_vocab = n_vocab

    def on_epoch_end(self, epoch, logs):
        if epoch % 2 != 0:
            return
        start = np.random.randint(0, len(self._data_x) - 1)
        pattern = self._data_x[start]
        print("Seed:")
        print(f'"{"".join([self._int_to_char[value] for value in pattern])}"')
        for i in range(100):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self._n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self._int_to_char[index]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1 : len(pattern)]
        print()


def read_text(seq_length=100) -> Tuple[np.array, np.array, Dict, List[int], int]:
    raw_text = open(FILENAME).read().lower()
    chars = sorted(list(set(raw_text)))

    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    n_chars = len(raw_text)
    n_vocab = len(chars)

    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i : i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)

    x = np.reshape(dataX, (n_patterns, seq_length, 1))
    x = x / float(n_vocab)

    y = np_utils.to_categorical(dataY)

    return x, y, int_to_char, dataX, n_vocab


def create_model(x_shape, y_shape) -> Model:
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_shape[1], x_shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y_shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main() -> None:
    x, y, int_to_char, data_x, n_vocab = read_text()
    checkpoint = ModelCheckpoint(
        FILENAME_PATTERN, monitor="loss", verbose=1, save_best_only=True, mode="min"
    )
    callbacks = [
        checkpoint,
        TextGeneratorCallback(int_to_char, data_x, n_vocab),
        TensorBoard(log_dir=f"logs/{datetime.now()}"),
    ]
    model = create_model(x.shape, y.shape)
    model.fit(x, y, epochs=20, batch_size=128, callbacks=callbacks)


if __name__ == "__main__":
    main()
