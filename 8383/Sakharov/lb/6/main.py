import tensorflow as tf 
device_name = tf.test.gpu_device_name()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import imdb
import re

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

print(type(data))
data = vectorize(data)
targets = np.array(targets).astype("float16")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
# Input - Layer
model.add(Dense(60, activation="relu", input_shape=(dim,)))
# Hidden - Layers
model.add(Dropout(0.4))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(50, activation="relu"))
# Output- Layer
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
print(np.mean(results.history["val_accuracy"]))

def plot_history(history, filename=""):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18,10))
    fig.suptitle('Loss and accuracy')

    ax1.plot(epochs, loss, color="green", label='Training loss')
    ax1.plot(epochs, val_loss, color = "blue", label='Validation loss')
    ax1.legend()

    ax2.plot(epochs, acc, color="green", label='Training acc')
    ax2.plot(epochs, val_acc, color = "blue", label='Validation acc')
    ax2.legend()

    plt.show()

    if (filename != ""):
        plt.savefig(filename)

def print_history(history):
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    print("T loss: {}; V loss: {}; T accuracy: {}; V accuracy: {}"
          .format(loss[-1], val_loss[-1], acc[-1], val_acc[-1]))
    plot_history(history)

print_history(results.history)
print(np.mean(results.history["val_accuracy"]))
read_text_from_input()