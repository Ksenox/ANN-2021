import numpy as np
from tensorflow.keras.models import Sequential
from keras import layers
from keras.datasets import imdb

idim = 10000

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def load_text(filename):
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
model.add(layers.Dense(50, activation="relu", input_shape=(idim, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

print(np.mean(history.history["val_accuracy"]))

while True:
    print("Input filename with review or 0 to stop")
    filename = input()
    if filename != "0":
        prediction = model.predict(load_text(filename))
        print(f"{filename} - {prediction}")
    else:
        break
