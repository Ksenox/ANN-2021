import matplotlib.pyplot as plt
import numpy as np
import re
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

def readTxt(filepath, maxSize=10000):
    f = open(filepath, 'r')
    txt = f.read()
    txt = re.sub(r"[^a-zA-Z0-9']", " ", txt)
    txt = txt.split(' ')
    index = imdb.get_word_index()
    resultCod = []
    result_l = []
    for word in txt:
        word = index.get(word)
        if word in range(1, maxSize):
            resultCod.append(word + 3)
    result_l.append(resultCod)
    return result_l



def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


size_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words= size_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])
#print(decoded)
data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

# Input - Layer
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(size_words, )))

# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))

# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))
#model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data = (test_x, test_y))

print(np.mean(results.history["val_accuracy"]))
test1 = readTxt("1.txt", 10000)
test1 = vectorize(test1)
result = model.predict(test1)
print("test 1")
print(result)
test2 = readTxt("2.txt", 10000)
test2 = vectorize(test2)
result = model.predict(test2)
print("test 2")
print(result)
test3 = readTxt("3.txt", 10000)
test3 = vectorize(test3)
result = model.predict(test3)
print("test 3")
print(result)
