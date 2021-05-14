import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb


def get_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def ensemble_predict(models, data, round=True):
    predictions = []
    for model in models:
        predictions.append(model.predict(data))
    predictions = np.asarray(predictions)
    predictions = np.mean(predictions, 0)
    if round:
        predictions = np.round(predictions)
    return predictions.flatten()


def ensemble_evaluate(models, data, targets):
    predictions = ensemble_predict(models, data)
    correct = (predictions == targets)
    accuracy = np.count_nonzero(correct) / len(targets)
    return accuracy


def load_file(filename):
    file = open(filename, 'r')
    text = file.read().lower()
    text = re.sub(r"[^a-z0-9' ]", "", text)
    text = text.split(" ")

    index = imdb.get_word_index()
    coded = [1]
    for word in text:
        num = index.get(word, 0)
        if num != 0:
            num += 3
        if num > top_words:
            num = 2
        coded.append(num)

    return coded


def predict_file(filename, models):
    data = load_file(filename)
    data = sequence.pad_sequences([data], maxlen=max_review_length)

    predictions = []
    for model in models:
        predictions.append(model.predict(data))
    result = np.mean(predictions)
    print(result)
    print('Positive' if result > 0.5 else 'Negative')


top_words = 10000
max_review_length = 500
embedding_vector_length = 32

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

X_test = data[:10000]
Y_test = targets[:10000]
X_train = data[10000:]
Y_train = targets[10000:]

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


model_count = 4
models = []
for i in range(model_count):
    train_len = len(Y_train) // model_count
    X_train_local = X_train[i * train_len: (i + 1) * train_len]
    Y_train_local = Y_train[i * train_len: (i + 1) * train_len]
    print(len(Y_train_local))

    models.append(get_model())
    models[i].fit(X_train_local, Y_train_local, validation_data=(X_test, Y_test), epochs=3, batch_size=64)

    score = models[i].evaluate(X_test, Y_test, verbose=0)[1]
    print("Accuracy: %.2f%%" % (score * 100))

print("Ensemble Accuracy: %.2f%%" % (ensemble_evaluate(models, X_test, Y_test) * 100))

predict_file('1.txt', models)
predict_file('2.txt', models)
predict_file('3.txt', models)
predict_file('4.txt', models)
predict_file('5.txt', models)
