import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from keras.models import Sequential
from keras.datasets import imdb
from tensorflow.python.keras.applications.densenet import layers


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


data = vectorize(data)
targets = np.array(targets).astype("float16")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

# Input - Layer
model = Sequential()
model.add(layers.Dense(60, activation="relu", input_shape=(10000,)))

# Hidden - Layers
model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.Dense(56, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(56, activation="relu"))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
print(np.mean(results.history["val_accuracy"]))


def draws(results):
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 3])
    plt.subplot(gs[0])
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(gs[1])
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def user_load():

    print("Input string: ")
    words = input()
    words = words.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
    dict_set = imdb.get_word_index()
    test_x = []
    test_y = []

    for word in words:
        if dict_set.get(word) in range(1, 10000):
            test_y.append(dict_set.get(word) + 3)
    test_x.append(test_y)

    print(test_x)
    test_x = vectorize(test_x)
    result = model.predict(test_x)
    print(result)



draws(results)
user_load()