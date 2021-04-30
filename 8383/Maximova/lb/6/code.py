#%matplotlib inline #отображает вывод в строке - доступно только для Jupyter
import matplotlib.pyplot as plt
import numpy as np
import string

from keras import models
from keras import layers
from keras.datasets import imdb

def loadDataIMBd(max_words):
    # нужно указать максимальное кол-во слов, исп-ое для анализа
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_words)
    data = np.concatenate((training_data, testing_data), axis=0)  # np.concatenate() - соед. массивы вдоль указанной оси
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    return data, targets

# подготовка данных
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # массив из 50к строк, длина каждой 10к
    for i, sequence in enumerate(sequences):         # индекс и значение
        results[i, sequence] = 1                     # 1 - число с данным индексом встречается, 0 - не встречается
    return results

def build_model():
    model = models.Sequential() #последовательная

    model.add(layers.Dense(50, activation='relu', input_shape=(10000, )))
    model.add(layers.Dropout(0.5, noise_shape=None, seed=None))

    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.4, noise_shape=None, seed=None))

    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))

    model.add(layers.Dense(1, activation='sigmoid'))  # бинарная классификация

    model.summary()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_loss(loss, val_loss, epochs):
    plt.plot(epochs, loss, label='Training loss', linestyle='--', linewidth=2, color="red")
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color="blue")
    plt.title('Training and Validation loss')  # оглавление на рисунке
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_acc(acc, val_acc, epochs):
    plt.clf()
    plt.plot(epochs, acc, label='Training accuracy', linestyle='--', linewidth=2, color="red")
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color="blue")
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def readFile(max_words):
    filename = ""
    while not filename:
        filename = input("Enter the name of the file to read: ")  # 1 - считываем текст
    text = getText(filename)  # 2 - создаем массив слов
    coded_text = vectorize(np.asarray([getIndexWord(text, max_words)]))  # 4 - векторизация
    return coded_text


def getText(filename):
    with open(filename) as inp:
        text = inp.read()
    text = wordProcessing(text)  # 3 - обрабатываем текст
    return text

def wordProcessing(text):
    text = text.lower()
    text = text.strip()
    for strp in string.punctuation:
        if strp in text:
            text = text.replace(strp, "")
    text = text.split(" ")

    # text = text.split()
    return text

def getIndexWord(text, maxDimension):
    # скачивание словаря, используемого для кодирования обзоров
    index = imdb.get_word_index()
    coded = []
    for word in text:
        indexWord = index.get(word)
        if indexWord is not None and indexWord < maxDimension:
            coded.append(indexWord + 3)
    return coded

max_words = 10000
data, targets = loadDataIMBd(max_words)

data = vectorize(data)
targets = np.array(targets).astype("float32")

test_x = data[:max_words]                           # разделение данных в другой пропорции
test_y = targets[:max_words]

train_x = data[max_words:]
train_y = targets[max_words:]
                                                    # создание и обучение модели
model = build_model()
history = model.fit(
    train_x, train_y,
    epochs=3,
    batch_size=1500,
    validation_data=(test_x, test_y)
)
print(np.mean(history.history["val_accuracy"]))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

plot_loss(loss, val_loss, epochs)
plot_acc(acc, val_acc, epochs)

# считывание отзыва
print("Do you want to enter your review? Enter y.")
answ = input()
if answ == 'y':
    coded_text = readFile(max_words)
prediction = model.predict(coded_text)
if prediction <= 0.5:
    print("Negative feedback")
else:
    print("Positive feedback")

# изучение датасета
# print("Categories:", np.unique(targets))
# print("Number of unique words:", len(np.unique(np.hstack(data))))
# length = [len(i) for i in data] #i - отзыв
# print("Average Review length:", np.mean(length))
# print("Standard Deviation:", round(np.std(length)))

# print("Label:", targets[0])
# print(data[0])

# скачивание словаря, используемого для кодирования обзоров
# index = imdb.get_word_index()
# reverse_index = dict([(value, key) for (key, value) in index.items()])
# decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])
# print(decoded)
