import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb
import re


num_words = 10000

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


data = vectorize(data, num_words)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]


model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(num_words, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))

print(np.mean(results.history["val_accuracy"]))
print("")


while True:
    text = input("Введите текст отзыва: ").lower()

    if text == 'exit':
        break

    text = re.sub(r"[^a-z0-9' ]", "", text)
    text = text.split(" ")

    coded = [1]
    for word in text:
        num = index.get(word, 0)
        if 1 <= num <= num_words:
            coded.append(num + 3)
        elif num > num_words:
            coded.append(2)

    coded = vectorize(np.asarray([coded]), num_words)
    result = model.predict(coded)[0][0]
    if result > 0.5:
        print("Положительный:", result)
    else:
        print("Отрицательный:", result)
    print("")
