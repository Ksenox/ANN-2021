import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
import re

# Представление данных в виде векторов
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
          results[i, sequence] = 1
    return results


# Чтение пользовательского текста
def read_txt(filepath, max_words=10000):
    f = open(filepath, 'r')
    txt = f.read().lower()
    txt = re.sub(r"[^a-zA-Z0-9']", " ", txt)  # убираем лишние символы
    txt = txt.split()  # разбиваем на слова
    index = imdb.get_word_index()  # загружаем словарь
    coded = [1]  # индекс начала последовательности
    coded.extend([index.get(i, 0) for i in txt])
    for i in range(len(coded)):
        if coded[i]:
            coded[i] += 3  # смещаем индексы
        if coded[i] >= max_words:
            coded[i] = 2  # отмечаем слова, не вошедшие в число max_words самых популярных
    return coded


# Загружаем данные
nwords = 1000  # столько уникальных слов будет загружено
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=nwords)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

# Обрабатываем данные
data = vectorize(data, nwords)
targets = np.array(targets).astype("float32")

test_x = data[:1000]
test_y = targets[:1000]
train_x = data[1000:]
train_y = targets[1000:]

# Создаем модель
model = models.Sequential()

# Input Layer
model.add(layers.Dense(50, activation="relu", input_shape=(nwords, )))

# Hidden Layers
model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.25, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))

# Output Layer
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(
 optimizer="adam",
 loss="binary_crossentropy",
 metrics=["accuracy"]
)

results = model.fit(
 train_x, train_y,
 epochs=2,
 batch_size=500,
 validation_data=(test_x, test_y)
)

print(np.mean(results.history["val_accuracy"]))

# Обрабатываем пользовательский текст
coded_txt = [read_txt("test1.txt", nwords)]
coded_txt.append(read_txt("test2.txt", nwords))
coded_txt.append(read_txt("test3.txt", nwords))
coded_txt.append(read_txt("test4.txt", nwords))

user_data = vectorize(coded_txt, nwords)
prediction = model.predict(user_data)
print(prediction)