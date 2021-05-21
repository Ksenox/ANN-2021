import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
import matplotlib.pyplot as plot


def plot_single_history(history, color="blue"):
    keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    titles = ["Loss", "Accuracy", "Val loss", "Val accuracy"]
    xlabels = ["epoch", "epoch", "epoch", "epoch"]
    ylabels = ["loss", "accuracy", "loss", "accuracy"]
    # ylims = [3, 1.1, 3, 1.1]

    for i in range(len(keys)):
        plot.subplot(2, 2, i + 1)
        plot.title(titles[i])
        plot.xlabel(xlabels[i])
        plot.ylabel(ylabels[i])
        # plot.gca().set_ylim([0, ylims[i]])
        plot.grid()
        values = history[keys[i]]
        plot.plot(range(1, len(values) + 1), values, color=color)


def make_third_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def make_second_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
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
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = text.split()

    index = imdb.get_word_index()
    coded = []
    for word in text:
        num = index.get(word, 0)
        if num != 0:
            num += 3
        if num <= top_words and num > 0:
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

test_len = round(len(data) * 0.8)
X_test = data[:test_len]
Y_test = targets[:test_len]
X_train = data[test_len:]
Y_train = targets[test_len:]

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

models = [
    make_second_model(),
    make_second_model(),
    make_third_model(),
    make_third_model()
]
for i in range(len(models)):
    model_count = len(models)

    h = models[i].fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=4, batch_size=64)

    score = models[i].evaluate(X_test, Y_test, verbose=0)[1]
    print("Accuracy: %.2f%%" % (score * 100))

    plot_single_history(h.history)

print("Ensemble Accuracy: %.2f%%" % (ensemble_evaluate(models, X_test, Y_test) * 100))

predict_file('good/1.txt', models)
predict_file('bad/1.txt', models)
