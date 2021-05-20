import string
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM, GRU
from keras.layers import Dropout

from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from tensorflow.keras.models import load_model

def loadDataIMDb(max_words):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_words)
    data = np.concatenate((training_data, testing_data), axis=0)  # соед. массивы вдоль указанной оси
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    return data, targets


def editData(data, targets, max_review_length):
    sep = (len(data) // 10) * 8

    X_train, Y_train = data[:sep], targets[:sep]  # 80 %
    X_test, Y_test = data[sep:], targets[sep:]  # 20 %

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    return X_train, Y_train, X_test, Y_test


def buildFirstModel(max_words, embedding_vector_length, max_review_length):
    model = Sequential()

    model.add(Embedding(input_dim=max_words,
                        output_dim=embedding_vector_length, input_length=max_review_length))
    # преобразование в плотные векторные представления
    model.add(LSTM(100))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def buildSecondModel(max_words, embedding_vector_length, max_review_length):
    # LSTM и сверточная НС
    model = Sequential()

    model.add(Embedding(input_dim=max_words,
                        output_dim=embedding_vector_length, input_length=max_review_length))
    # преобразование в плотные векторные представления
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(50))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def buildThirdModel(max_words, embedding_vector_length, max_review_length):
    model = Sequential()

    model.add(Embedding(input_dim=max_words,
                        output_dim=embedding_vector_length, input_length=max_review_length))
    # преобразование в плотные векторные представления
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.35))
    model.add(LSTM(32))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def printAcc(model):
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

def plotLoss(loss, val_loss, epochs, numb):
    plt.plot(epochs, loss, label="Training loss", linestyle='--', linewidth=2, color='red')
    plt.plot(epochs, val_loss, "b", label="Validation loss", color='blue')

    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()
    plt.savefig('Gr/Loss_Model_' + str(numb) + '.png', format="png", dpi=240)
    plt.show()

def plotAcc(acc, val_acc, epochs, numb):
    plt.clf()
    plt.plot(epochs, acc, label="Training accuracy", linestyle='--', linewidth=2, color='red')
    plt.plot(epochs, val_acc, "b", label="Validation accuracy", color='blue')

    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid()
    plt.savefig('Gr/Accuracy_Model_' + str(numb) + '.png', format="png", dpi=240)
    plt.show()

def fitModels(all_models, X_train, Y_train, X_test, Y_test):
    numb = 1
    for model in all_models:
        history = model.fit(
            X_train, Y_train,
            epochs=2,
            batch_size=64,
            validation_data=(X_test, Y_test))  # 0.2
        printAcc(model)

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        epochs = range(1, len(loss) + 1)

        plotLoss(loss, val_loss, epochs, numb)
        plotAcc(acc, val_acc, epochs, numb)
        model.save('Models/INS_' + str(numb) + '.h5')
        numb = numb + 1

def ensemble(validation_data):
    model1 = load_model("Models/INS_1.h5")
    model2 = load_model("Models/INS_2.h5")
    model3 = load_model("Models/INS_3.h5")

    preds_a = model1.predict(validation_data)
    preds_b = model2.predict(validation_data)
    preds_c = model3.predict(validation_data)

    final_preds = (preds_a + preds_b + preds_c) / 3
    return final_preds

def calcAcc(final_preds, Y_test):
    right_predict = 0
    all_predict = len(Y_test)

    for i in range(len(final_preds)):
        if(final_preds[i] > 0.5):
            final_preds[i] = 1
        else:
            final_preds[i] = 0
        if(final_preds[i] == Y_test[i]):
            right_predict = right_predict + 1
    return right_predict / all_predict

def readFile(max_words, max_review_length):
    filename = ""
    while not filename:
        filename = input("Enter the name of the file to read: ")  # 1 - считываем текст
    text = getText(filename)  # 2 - создаем массив слов
    coded_text = np.array([getIndexWord(text, max_words)])
    return sequence.pad_sequences(coded_text, maxlen=max_review_length)

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
max_review_length = 500
embedding_vector_length = 32

data, targets = loadDataIMDb(max_words)
targets = np.array(targets).astype("float32")
X_train, Y_train, X_test, Y_test = editData(data, targets, max_review_length)

# создаем модель - обучаем - выводим точность - рисуем - сохраняем
# соединяем все модели в ансамбль

all_models = []
all_models.append(buildFirstModel(max_words, embedding_vector_length, max_review_length))
all_models.append(buildSecondModel(max_words, embedding_vector_length, max_review_length))
all_models.append(buildThirdModel(max_words, embedding_vector_length, max_review_length))

print("Нажмите 1, если хотите получить предсказание по каждой из модели")
print("Нажмите 2, если хотите получить предсказание с помощью ансамбля")
print("Нажмите 3, если хотите загрузить собственный текст")

answer = input()

if answer == "1":
    fitModels(all_models, X_train, Y_train, X_test, Y_test) # обучить модели еще раз
if answer == "2":
    final_preds = ensemble(X_test)
    print("Accuracy ensemble: %s %%" % (calcAcc(final_preds, Y_test) * 100))
if answer == "3":
    final_preds = ensemble(readFile(max_words, max_review_length))
    if final_preds > 0.5:
        print("Ensemble prediction: ", final_preds[0][0])
        print("Positive")
    else:
        print("Ensemble prediction: ", final_preds[0][0])
        print("Negative")
