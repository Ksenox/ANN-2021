import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from keras import layers
from keras import models
from keras.datasets import imdb

INDEX = imdb.get_word_index()


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results


def get_data() -> Tuple[np.array, np.array]:
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    return data, targets


def covert_text(text: str, max_size=10000) -> np.array:
    result = []

    for word in text.split():
        index = INDEX.get(word.lower())
        if index is None or index + 3 > max_size:
            continue
        result.append(index + 3)

    reverse_index = dict([(value, key) for (key, value) in INDEX.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in result])
    print(decoded)

    return vectorize([result], max_size)


def main(file_path: Path):
    capacity = 10000
    input_data = covert_text(file_path.read_text())
    data, targets = get_data()

    data = vectorize(data, capacity)
    targets = np.array(targets).astype("float32")

    train_x, train_y = data[10000:], targets[10000:]
    test_x, test_y = data[:10000], targets[:10000]

    model = models.Sequential()

    # Input - Layer
    model.add(layers.Dense(100, activation="relu", input_shape=(capacity,)))
    # Hidden - Layers
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    results = model.fit(
        train_x, train_y,
        epochs=2,
        validation_data=(test_x, test_y)
    )

    print(np.mean(results.history["val_accuracy"]))

    print(model.predict(input_data))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
