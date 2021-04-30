import numpy as np
from keras import models, Sequential
from keras import layers
import re
from keras.datasets import imdb


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def convertString(str):
    # удаляем все символы кроме букв, пробельных символов и апострофа
    # приводим к нижнему регистру и разделяем слова чере пробел
    str = re.sub('[^A-Za-z\s\']', '', str).lower().split(' ')
    index = imdb.get_word_index()   # послучаем словарь слов и их индексов
    list = []
    for i in str:
        if i in index.keys():
            list.append(index[i] + 3)   # преобразуем слова в индексы
    return list


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

# for i in range(len(data)):
#     for j in range(len(data[i])):
#         if data[i][j] >= 8000:
#             data[i][j] = 0

data = vectorize(data)
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()
# Input - Layer
model.add(layers.Dense(50, activation = 'relu'))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = 'relu'))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = 'relu'))
# Output- Layer
model.add(layers.Dense(1, activation = 'sigmoid'))
# model.summary()

model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 500,
 validation_data = (test_x, test_y)
)

print("Validation accuracy: ", np.mean(results.history['val_accuracy']))


while True:

    str = input("Enter your string or enter \"exit\" to out: ")

    if str == 'exit': break

    str = convertString(str)
    if len(str) == 0:
        print("Words is undefined")
        continue
    for i in str:
        if i > 10000: str.remove(i)

    if 0.5 < model.predict(vectorize(np.asarray([str]))):
        print("I predict it is positive feedback")
    else:
        print("I predict it is negative feedback")


