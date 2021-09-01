import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from tensorflow.keras import datasets
import string
dimension = 10000

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def load_user_input(text, target):
    def genNum(data, dic):
        data = data.translate(str.maketrans(dict.fromkeys(string.punctuation))).split()
        for i in range(len(data)):
            num = dic.get(data[i])
            if (num == None):
                data[i] = 0
            else:
                data[i] = num
        return data

    dic = dict(datasets.imdb.get_word_index())
    test_x = []
    test_y = np.array(target).astype("float32")
    for i in range(0, len(text)):
        test_x.append(genNum(text[i], dic))
    test_x = vectorize(test_x)
    return test_x, test_y

x = [
    "Amazing film love this genre",
    "Storytelling felt primitive and kinda forced",
    "Absolutely astonishing bravo",
    "Could not cope with pacing of the plot just bad",
    "This artwork did a bare minimum that is not enough"
]
y = [1, 0, 1, 0, 0]

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
data = vectorize(data, dimension)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = models.Sequential()

model.add(layers.Dense(50, activation="relu", input_shape=(dimension,)))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

test_x, test_y = load_user_input(x, y)
custom_loss, custom_acc = model.evaluate(test_x, test_y)
print('custom_acc:', custom_acc)
preds = model.predict(test_x)
plt.figure(3, figsize=(8,5))
plt.title("Custom dataset predications")
plt.plot(test_y, 'r', marker='v', label='truth')
plt.plot(preds, 'b', marker='x', label='pred')
plt.legend()
plt.show()
plt.clf()

plt.figure(1, figsize=(8, 5))
plt.title("Training and test accuracy")
plt.plot(h.history['accuracy'], 'r', label='train')
plt.plot(h.history['val_accuracy'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()

#@title

plt.figure(1, figsize=(8, 5))
plt.title("Training and test loss")
plt.plot(h.history['loss'], 'r', label='train')
plt.plot(h.history['val_loss'], 'b', label='test')
plt.legend()
plt.show()
plt.clf()

#@title

print(np.mean(h.history["val_accuracy"]))
print(np.mean(preds))