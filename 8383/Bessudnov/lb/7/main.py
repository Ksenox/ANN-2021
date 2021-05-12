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


def generateModel1():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(64, dropout=0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generateModel2():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(LSTM(32, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

top_words = 20000
trainSize = len(data) * 8 // 10
embedding_vector_length = 32
max_review_length = 500
modelsCount = 4
trainBlockSize = trainSize // modelsCount
testBlockSize = (len(data) - trainSize) // modelsCount

X_train = data[:trainSize]
Y_train = targets[:trainSize]
X_test = data[trainSize:]
Y_test = targets[trainSize:]

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

scores = []

for i in range(modelsCount):
    X_block_train = X_train[i * trainBlockSize : (i + 1) * trainBlockSize]
    Y_block_train = Y_train[i * trainBlockSize : (i + 1) * trainBlockSize]
    X_block_test = X_test[i * testBlockSize : (i + 1) * testBlockSize]
    Y_block_test = Y_test[i * testBlockSize : (i + 1) * testBlockSize]

    if (i % 2 == 0):
        print("type 1")
        model = generateModel1()
    else:
        print("type 2")
        model = generateModel2()

    model.summary()
    model.fit(X_block_train, Y_block_train, validation_data=(X_block_test, Y_block_test), epochs=2, batch_size=16)
    score = model.evaluate(X_block_test, Y_block_test, verbose=0)
    print(score)
    scores.append(score[1])
    # model.save("model_lb7_" + str(i % 2) + "_" + str(i // 2) + ".h5")


print("Accuracy: %.2f%%" % (np.mean(scores)*100))