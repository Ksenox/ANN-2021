import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow.python.keras.layers.core import Dropout


def readText(file):
    text = ""
    with open(file, 'r') as fd:
        text = fd.read()
    result = []
    index = imdb.get_word_index()
    for w in text.split():
        i = index.get(w.lower())
        if i is not None and i < 10000:
            result.append(i+3)
    result = np.array([result])
    result = sequence.pad_sequences(result, maxlen=500)
    return result
    # return vectorize([result], 10000)


def createModel1():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length,
                        input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(64, dropout=0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def createModel2():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length,
                        input_length=max_review_length))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(LSTM(32, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def fitModels(models):
    scores = []
    for i, model in enumerate(models):
        X_block_train = X_train[i * trainBlockSize: (i + 1) * trainBlockSize]
        Y_block_train = Y_train[i * trainBlockSize: (i + 1) * trainBlockSize]
        X_block_test = X_test[i * testBlockSize: (i + 1) * testBlockSize]
        Y_block_test = Y_test[i * testBlockSize: (i + 1) * testBlockSize]
        model.summary()
        model.fit(X_block_train, Y_block_train, validation_data=(
            X_block_test, Y_block_test), epochs=2, batch_size=16)
        score = model.evaluate(X_block_test, Y_block_test, verbose=0)
        print(score)
        scores.append(score[1])
    return scores


def predictModels(models, filename):
    scores = []
    text = readText(filename)
    for i in range(len(models)):
        scores.append(models[i].predict(text))
        # scores[i] = models[i].predict(text)
    return scores


(training_data, training_targets), (testing_data,
                                    testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

top_words = 10000
trainSize = len(data) * 8 // 10
embedding_vector_length = 32
max_review_length = 500


X_train = data[:trainSize]
Y_train = targets[:trainSize]
X_test = data[trainSize:]
Y_test = targets[trainSize:]

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


models = [createModel1(), createModel2()]
trainBlockSize = trainSize // len(models)
testBlockSize = (len(data) - trainSize) // len(models)

fitScores = fitModels(models)
print("Accuracy: %.2f%%" % (np.mean(fitScores)*100))

predScores = predictModels(models, "1")
print(predScores)
predScores = predictModels(models, "2")
print(predScores)
predScores = predictModels(models, "3")
print(predScores)
predScores = predictModels(models, "4")
print(predScores)
