import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import imdb

dim = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dim)
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
print([i for i in data[0]])
decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])

print(decoded)


def vectorize(sequences, dimension=dim):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


print(type(data))
data = vectorize(data)
targets = np.array(targets).astype("float16")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
model.add(Dense(70, activation="relu", input_shape=(dim,)))
model.add(Dropout(0.4))
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(40, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
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
    plt.plot(epochs, val_loss, 'p--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(gs[1])
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'p--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


draws(results)


def read_text_from_input():
    dictionary = imdb.get_word_index()

    words = input()
    words = re.sub(r"[^\w']", " ", words).split(' ')

    valid = []
    for word in words:
        word = dictionary.get(word)
        if word in range(1, dim):
            valid.append(word + 3)

    X = []
    X.append(valid)
    print(X)
    X = vectorize(X)
    result = model.predict(X)
    print(result)


read_text_from_input()
