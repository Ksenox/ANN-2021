import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, Sequential
from tensorflow.keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

size = 10000

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_text(file):
    str = ""
    with open(file, 'r') as fd:
        str = fd.read()
    result = []
    index = imdb.get_word_index()

    for w in str.split():
        i = index.get(w.lower())
        if i is not None and i < size:
            result.append(i)

    return vectorize([result], size)


data = vectorize(data, size)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]


model = Sequential()
# Input - Layer
model.add(layers.Dense(50, activation="relu", input_shape=(size, )))
# Hidden - Layers

model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
 optimizer="adam",
 loss="binary_crossentropy",
 metrics=["accuracy"]
)

results = model.fit(
 train_x, train_y,
 epochs=4,
 batch_size=500,
 validation_data=(test_x, test_y)
)

print(np.mean(results.history["val_accuracy"]))

print(model.predict(load_text("1")))  # 10
print(model.predict(load_text("2")))  # 1
print(model.predict(load_text("3")))  # 5
print(model.predict(load_text("4")))  # 9
