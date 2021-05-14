# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow import metrics
import re

top_words = 20000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vecor_length = 32

count_models = 4
num_of_predicts = 100
models = []

for i in range(0, count_models):
    x_train = X_train[i * int(len(X_train) / count_models): (i + 1) * int(len(X_train) / count_models)]
    y_train = Y_train[i * int(len(Y_train) / count_models): (i + 1) * int(len(Y_train) / count_models)]

    models.append(Sequential())

    models[i].add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    models[i].add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    models[i].add(MaxPooling1D(pool_size=2))
    models[i].add(Dropout(0.2))
    models[i].add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    models[i].add(Dropout(0.2))
    models[i].add(Dense(1, activation='sigmoid'))

    models[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    models[i].fit(x_train, y_train, epochs=3, batch_size=64)

#-------------------------------Ансамблирование-------------------------------

def ansambl(models, data):
    count = []
    for i in range(0, len(data)):
        predictes = []
        for m in models:
            predictes.append(m.predict(np.asarray([data[i]])))
        count.append(1 if sum(predictes) / len(predictes) > 0.5 else 0)
        if i % 10 == 0: print(i)
    return count

ind = 0

for m in models:
    scores = m.evaluate(X_test[0:num_of_predicts], Y_test[0:num_of_predicts])
    ind += 1
    print("Model " + str(ind) + " accuracy: %.2f%%" % (scores[1]*100))

count = ansambl(models, X_test[0:num_of_predicts])
res = np.asarray(count).reshape(num_of_predicts, 1)
met = metrics.Accuracy()
met.update_state(res, Y_test[0:num_of_predicts].reshape(num_of_predicts, 1))
print("Models accuracy: %.2f%%" % (met.result().numpy()*100))

#-------------------------------Анализ введенного текста----------------------

def convertString(str):
    # удаляем все символы кроме букв, пробельных символов и апострофа
    # приводим к нижнему регистру и разделяем слова чере пробел
    str = re.sub('[^A-Za-z\s\']', '', str).lower().split(' ')
    index = imdb.get_word_index()  # послучаем словарь слов и их индексов
    list = []
    for i in str:
        if i in index.keys():
            list.append(index[i] + 3)  # преобразуем слова в индексы
    return list


while True:

    str = input("Enter your string or enter \"exit\" to out: ")
    print(str)

    if str == 'exit': break

    str = convertString(str)

    if len(str) == 0:
        print("Words is undefined")
        continue
    for i in str:
        if i > 10000: str.remove(i)

    predict = ansambl(models, [str])

    if 0.5 < predict[0]:
        print("I predict it is positive feedback")
    else:
        print("I predict it is negative feedback")