import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import string
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results

def textFromFile(fileName, dimension=10000):
    file = open(fileName, "r")
    text = file.read()
    file.close()
    text.lower()
    tt = str.maketrans(dict.fromkeys("!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"))
    text = text.translate(tt).split()
    index = imdb.get_word_index()
    codedText = []
    for word in text:
        i = index.get(word)
        if i is not None and i < dimension:
            codedText.append(i+3)
    codedText = vectorize(np.asarray([codedText]))
    return codedText


dimension=10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
#
# print("Categories:", np.unique(targets))
# print("Number of unique words:", len(np.unique(np.hstack(data))))
# length = [len(i) for i in data]
# print("Average Review length:", np.mean(length))
# print("Standard Deviation:", round(np.std(length)))
# print("Label:", targets[0])
# print(data[0])

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]])
# print(decoded)
data = vectorize(data, dimension)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
print(data[0])


model = models.Sequential()
# Input - Layer
model.add(layers.Dense(50, activation="relu", input_shape=(dimension, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

print(np.mean(history.history["val_accuracy"]))


while(1):
    print("Print file name *.txt or stop")
    fileName = input()
    if fileName == "stop":
        break
    text = textFromFile(fileName, dimension)
    res = model.predict(text)
    if res >= 0.5:
        print("Good")
    else:
        print("Bad")
