import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from keras.models import Sequential
from keras.datasets import imdb
from tensorflow.python.keras.applications.densenet import layers

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))
print("Label:", targets[0])
print(data[0])
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])
print(decoded)


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
model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))

# Hidden - Layers

model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

print(np.mean(results.history["val_accuracy"]))


def draws(H):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 3])
    plt.subplot(gs[0])
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(gs[1])
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


draws(results)


def textInput():
    words = input().split(' ')

    imdb_dict = imdb.get_word_index()
    words_rate = []
    tmp = []

    words_range = 1000
    for word in words:
        if imdb_dict.get(word) in range(1, words_range):
            tmp.append(imdb_dict.get(word) + 3)
    words_rate.append(tmp)

    print(words_rate)
    words_rate = vectorize(words_rate)
    result = model.predict(words_rate)
    print(result)


if __name__ == '__main__':
    textInput()
