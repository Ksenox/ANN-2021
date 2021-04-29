import matplotlib.pyplot as plot
import numpy as np
from keras import Sequential
from keras import layers
from keras.utils import to_categorical
from keras import models
from keras.datasets import imdb


def vectorize(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_text(filename, idim):
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data += [w.strip(''.join(['.', ',', ':', ';', '!', '?', '(', ')'])).lower() for w in line.strip().split()]
    index = imdb.get_word_index()
    x_test = []
    for w in data:
        if w in index and index[w] < idim:
            x_test.append(index[w])
    x_test = vectorize([np.array(x_test)], idim)
    return x_test


def plot_single_history(history, color="blue"):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    titles = ["Loss", "Accuracy", "Val loss", "Val accuracy"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "accuracy", "loss", "accuracy"]
    # ylims = [3, 1.1, 3, 1.1]

    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        # plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(1, len(values) + 1), values, color=color)


idim = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=idim)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
data = vectorize(data, idim)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(idim,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.45))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_x, train_y, epochs=5, batch_size=3000, validation_data=(test_x, test_y))

plot_single_history(history.history)
plot.show()

results = model.evaluate(test_x, test_y)
print(results)

goodFiles = [
    "good/1.txt"
]
badFiles = [
    "bad/1.txt"
]

print("Good:")
for filePath in goodFiles:
    prediction = model.predict(load_text(filePath, idim))
    print(prediction)

print("Bad:")
for filePath in badFiles:
    prediction = model.predict(load_text(filePath, idim))
    print(prediction)
