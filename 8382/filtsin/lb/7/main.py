import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout, SimpleRNN, GRU
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

num_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

train_length = (data.shape[0] // 10) * 8

X_train = data[:train_length]
Y_train = targets[:train_length]
X_test = data[train_length:]
Y_test = targets[train_length:]

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vecor_length = 32
model1 = Sequential()
model1.add(Embedding(num_words, embedding_vecor_length, input_length=max_review_length))
model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(LSTM(100))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model1.summary())

model1.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)
scores = model1.evaluate(X_test, Y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

model2 = Sequential()
model2.add(Embedding(num_words, embedding_vecor_length, input_length=max_review_length))
model2.add(SimpleRNN(32, return_sequences=True))
model2.add(SimpleRNN(32))
model2.add(Dense(1, activation="sigmoid"))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.summary())

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model2.summary())

model2.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)
scores = model2.evaluate(X_test, Y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

models = [model1, model2]


def ensemble_prediction(data):
    predictions = np.array([])
    for model in models:
        predictions = np.append(predictions, model.predict(data))
    predictions = np.mean(predictions, 0)
    return predictions.flatten()


def ensemble_accuracy(prediction, label):
    return np.count_nonzero(np.round(prediction) == label) / len(label)


print("Ensemble accuracy: %.2f%%" % ensemble_accuracy(ensemble_prediction(X_test), Y_test))


def load_text(file):
    str = ""
    with open(file, 'r') as fd:
        str = fd.read()
    result = []
    index = imdb.get_word_index()

    for w in str.split():
        i = index.get(w.lower())
        if i is not None and i < num_words:
            result = np.append(result, i + 3)

    return result


def prediction_text(file):
    vec = load_text(file)
    pad = sequence.pad_sequences([vec], maxlen=max_review_length)
    return ensemble_prediction(pad)


print(prediction_text("1"))
print(prediction_text("2"))
print(prediction_text("3"))
print(prediction_text("4"))
