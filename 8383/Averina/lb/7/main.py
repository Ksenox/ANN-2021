import re

import numpy as np
from keras import layers
from keras.datasets import imdb
from keras_preprocessing import sequence
from numpy import vectorize
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, MaxPooling1D, Conv1D, Dropout
from tensorflow.python.keras.models import Sequential

from keras.datasets import imdb


top_words = 20000
max_review_length = 500
m = 4

def create():

    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    train_size = len(data) * 8 // 10
    embedding_vector_length = 32


    X_train = data[:train_size]
    Y_train = targets[:train_size]
    X_test = data[train_size:]
    Y_test = targets[train_size:]

    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # model = Sequential()
    # model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    #
    # model.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=3, batch_size=64)
    # print(model.evaluate(X_train, Y_train, verbose=0))


    models = []
    scores = []


    for i in range(m):
        x_train = X_train[int(len(X_train) / m) * i: int(len(X_train) / m) * (i + 1)]
        y_train = Y_train[int(len(Y_train) / m) * i: int(len(Y_train) / m) * (i + 1)]

        models.append(Sequential())
        models[i].add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        models[i].add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        models[i].add(MaxPooling1D(pool_size=2))
        models[i].add(Dropout(0.2))
        models[i].add(LSTM(100))
        models[i].add(Dropout(0.2))
        models[i].add(Dense(1, activation='sigmoid'))

        models[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        models[i].fit(x_train, y_train, validation_data=(x_train, y_train), epochs=3, batch_size=64)
        print(models[i].evaluate(x_train, y_train, verbose=0))
        scores.append(models[i].evaluate(x_train, y_train, verbose=0)[1])

    print("Accuracy: %.2f%%" % (np.mean(scores)*100))
    return models

def convert_text(text):
    text = re.sub(r'[^\w\s]', ' ', text).lower().split(' ')
    indexes = imdb.get_word_index()
    text_indexes = []
    for word in text:
        if word in indexes.keys():
            ind = indexes[word]
            if ind < top_words:
                text_indexes.append(ind + 3)
    return sequence.pad_sequences([text_indexes], maxlen=max_review_length)


def read_text_from_file(num):
    with open('./test/test_' + str(num) + '.txt', 'r') as file:
        return file.read()


def main():
    models = create()
    answers = [1, 0, 0, 1]
    for i in range(1, 5):
        text = read_text_from_file(i)
        index_text = convert_text(text)
        prediction = []
        for model in models:
            prediction.append(model.predict(index_text))
        print(np.mean(prediction))
        prediction = round(np.mean(prediction))
        print('It is positive feedback') if prediction else print('It is negative feedback')
        print('Answer is true.') if prediction == answers[i - 1] else print('Answer is false.')
        print('\n')


if __name__ == '__main__':
    main()
