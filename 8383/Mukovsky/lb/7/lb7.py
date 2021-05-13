import numpy as np
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score
import re
from keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

X_test = data[:10000]
Y_test = targets[:10000]
X_train = data[10000:]
Y_train = targets[10000:]

max_review_length = 500
embedding_vector_length = 32
top_words = 10000

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


def get_model_a():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.4))
    model.add(LSTM(40, return_sequences=True, dropout=0.5))
    model.add(LSTM(20, dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_model_b():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_model_c():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.3))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


train_size, test_size = len(X_train) // 3, len(X_test) // 3


def get_ensemble_predictions(all_models, x_test, X):
    result = []
    for m in all_models:
        if X:
            print(m.predict(x_test, verbose=0))
        result.append(np.round(m.predict(x_test, verbose=0)))
    result = np.asarray(result)
    result = [np.round(np.mean(x)) for x in zip(*result)]
    return np.asarray(result).astype('int')


all_models = [get_model_a(), get_model_b(), get_model_c()]

for i, model in enumerate(all_models):
    x_train = X_train[i * train_size: (i + 1) * train_size]
    y_train = Y_train[i * train_size: (i + 1) * train_size]
    x_test = X_test[i * test_size: (i + 1) * test_size]
    y_test = Y_test[i * test_size: (i + 1) * test_size]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=64)
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("model accuracy: %.2f%%" % (scores[1] * 100))

result_predictions = get_ensemble_predictions(all_models, X_test, False)
acc = accuracy_score(Y_test, result_predictions)
print("ensemble accuracy: %.2f%%" % (acc * 100))


def read_text_from_input():
    dictionary = imdb.get_word_index()
    words = input()
    words = re.sub(r"[^\w']", " ", words).split(' ')

    valid = []
    for word in words:
        word = dictionary.get(word)
        if word in range(1, 10000):
            valid.append(word + 3)

    X = []
    X.append(valid)
    result = sequence.pad_sequences(X, maxlen=max_review_length)
    print(get_ensemble_predictions(all_models, result, True))


read_text_from_input()
