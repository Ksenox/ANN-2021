import numpy as np
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import accuracy_score


def read_txt(filepath, maxSize, models):
    f = open(filepath, 'r')
    txt = f.read()
    txt = re.sub(r"[^a-zA-Z0-9']", " ", txt)
    txt = txt.split(' ')
    index = imdb.get_word_index()
    resultCod = []
    result_l = []
    for word in txt:
        word = index.get(word)
        if word in range(1, maxSize):
            resultCod.append(word + 3)
    result_l.append(resultCod)
    result_l = sequence.pad_sequences(result_l, maxlen=500)
    result = prediction(models, result_l)
    if result:
        print("good")
        print(result)
    else:
        print("bad")
        print(result)



def prediction(models, x_test):
    yPredict = []
    for i, model in enumerate(models):
        res = model.predict(x_test)
        #print(res)
        yPredict.append(np.round(res))
    yPredict = np.asarray(yPredict)
    predictions = np.round(np.mean(yPredict, 0))
    return predictions


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

max_review_length = 500
X_train = sequence.pad_sequences(train_x, maxlen=max_review_length)
X_test = sequence.pad_sequences(test_x, maxlen=max_review_length)

top_words = 10000
embedding_vecor_length = 32

# model a

model_a = Sequential()
model_a.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model_a.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_a.add(layers.MaxPooling1D(pool_size=2))
model_a.add(LSTM(100))
model_a.add(Dense(1, activation='sigmoid'))

# model b
model_b = Sequential()
model_b.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model_b.add(layers.Flatten())
model_b.add(Dense(50, activation="relu"))
model_b.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model_b.add(Dense(50, activation="relu"))
model_b.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model_b.add(Dense(50, activation="relu"))
model_b.add(layers.Dense(1, activation="sigmoid"))

# model c
model_c = Sequential()
model_c.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model_c.add(layers.GRU(70, recurrent_dropout=0.2))
model_c.add(Dense(1, activation="sigmoid"))

# model d
model_d = Sequential()
model_d.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model_d.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
model_d.add(layers.MaxPooling1D())
model_d.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
model_d.add(layers.MaxPooling1D())
model_d.add(layers.Dropout(0.2))
model_d.add(layers.Flatten())
model_d.add(Dense(50, activation='relu'))
model_d.add(Dense(1, activation='sigmoid'))

models = [model_a, model_b, model_c, model_d]
train_batch_len = len(train_y) // 4

for i in range(4):
    print("Model №", i + 1)
    x_train = X_train[i * train_batch_len:(i + 1) * train_batch_len]
    y_train = train_y[i * train_batch_len:(i + 1) * train_batch_len]
    models[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models[i].fit(x_train, y_train, validation_data=(X_test, test_y), epochs=2, batch_size=64)
    scores = models[i].evaluate(X_test, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

y_pred = prediction(models, X_test)
acc = accuracy_score(test_y, y_pred)
print("Ensemble accuracy : %.2f%%" % (acc*100))

print("test 1")
read_txt("1.txt", 10000, models)
print("test 2")
read_txt("2.txt", 10000, models)
print("test 3")
read_txt("3.txt", 10000, models)

