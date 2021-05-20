import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import re


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
          results[i, sequence] = 1
    return results



def read_txt(filepath, idim=10000):
    f = open(filepath, 'r')
    txt = f.read().lower()
    txt = re.sub(r"[^a-z0-9']", " ", txt)
    print(txt)
    txt = txt.split()

    index = imdb.get_word_index()

    coded = [-2]
    coded.extend([index.get(i, 0) for i in txt])
    for i in range(len(coded)):
        if coded[i]:
            coded[i] += 3
        if coded[i] >= idim:
            coded[i] = 2
    print(coded)
    return coded



num = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)



data = vectorize(data, num)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = models.Sequential()

model.add(layers.Dense(50, activation="relu", input_shape=(num, )))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.35))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
H = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y), verbose=0)

print("Точность:")
print(np.mean(H.history["val_accuracy"]))


for i in range(4):
    name = str(i+1) + ".txt"
    user_data = [read_txt(name, num)]
    user_data = vectorize(user_data, num)
    prediction = model.predict(user_data)
    if prediction >= 0.5:
        print("Good")
    else:
        print("Bad")
    print(prediction)
    print("\n")
