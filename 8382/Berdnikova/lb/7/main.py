import tensorflow as tf

device_name = tf.test.gpu_device_name()

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

X_test = data[:10000]
Y_test = targets[:10000]
X_train = data[10000:]
Y_train = targets[10000:]

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32

model_lstm = Sequential([
    Embedding(10000, embedding_vector_length, input_length=max_review_length),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")])

model_cnn = Sequential([
    Embedding(10000, embedding_vector_length, input_length=max_review_length),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.4),
    LSTM(128),
    Dropout(0.3),
    Dense(1, activation='sigmoid')])

model_cnl = Sequential([
    Embedding(10000, embedding_vector_length, input_length=max_review_length),
    Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPool1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(128),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

train_size = len(X_train) // 3
test_size = len(X_test) // 3
models = [model_lstm, model_cnn, model_cnl]

for i, mod in enumerate(models):
    x_train = X_train[i * train_size: (i + 1) * train_size]
    y_train = Y_train[i * train_size: (i + 1) * train_size]

    x_test = X_test[i * test_size: (i + 1) * test_size]
    y_test = Y_test[i * test_size: (i + 1) * test_size]

    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    mod.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=64)
    scores = mod.evaluate(x_test, y_test, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def forecast(models, x_test, load):
    combo = []

    for i, m in enumerate(models):
        if load:
            print(m.predict(x_test, verbose=0))
        combo.append(np.round(m.predict(x_test, verbose=0)))

    combo = np.asarray(combo)
    combo = np.round(np.mean(combo, 0))
    return combo


combo = forecast(models, X_test, False)
rating = accuracy_score(Y_test, combo)
print("Accuracy: %.2f%%" % (rating * 100))


def user_load():
    print("Input string: ")
    words = input()
    words = words.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
    dict_set = imdb.get_word_index()
    test_x = []
    test_y = []

    for word in words:
        if dict_set.get(word) in range(1, 10000):
            test_y.append(dict_set.get(word) + 3)
    test_x.append(test_y)
    print(test_x)
    result = sequence.pad_sequences(test_x, maxlen=max_review_length)
    print(forecast(models, result, True))


user_load()