import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence

def get_vec_from_file(file):
    with open(file, 'r') as txtFile:
        text = txtFile.read()
    words = text_to_word_sequence(text)
    vocabulary = imdb.get_word_index()
    x_predict = []
    for i in range(len(words)):
        if words[i] not in vocabulary:
            continue
        if vocabulary[words[i]] + 3 < 10000:
            x_predict.append(vocabulary[words[i]] + 3)
    return sequence.pad_sequences([x_predict], maxlen=max_review_length)


def predict_sample_text(pred):
    predictions = []
    for i in range(m):
        predictions.append(models[i].predict(pred))
    prediction = ((np.mean(predictions)) > 0.5).astype("int32")
    print("Mean predictions %.2f%%" % (np.mean(predictions) * 100))
    if prediction == 1:
        print("Good")
    else:
        print("Bad")

max_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_words)
train_x = np.concatenate((training_data, testing_data), axis=0)
train_y = np.concatenate((training_targets, testing_targets), axis=0)
train_y = np.array(train_y).astype("float32")

max_review_length = 500
train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
embedding_vector_length = 32

model1 = Sequential()
model1.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
model1.add(LSTM(100))
model1.add(Dense(1, activation='sigmoid'))

model2 = Sequential()
model2.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
model2.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(LSTM(100))
model2.add(Dense(1, activation='sigmoid'))

model3 = Sequential()
model3.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
model3.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Dropout(0.2))
model3.add(LSTM(100))
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid'))

models = []
scores = []
m = 4
for i in range(m):
    x_train = train_x[int(len(train_x) / m) * i: int(len(train_x) / m) * (i + 1)]
    y_train = train_y[int(len(train_y) / m) * i: int(len(train_y) / m) * (i + 1)]

    models.append(model3)
    models[i].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models[i].fit(x_train, y_train, validation_data=(x_train, y_train), epochs=3, batch_size=64)
    print(models[i].evaluate(x_train, y_train, verbose=0))
    scores.append(models[i].evaluate(x_train, y_train, verbose=0)[1])


print("Mean accuracy: %.2f%%" % (np.mean(scores) * 100))

while True:
    print("Input filename with review or 0 to stop")
    filename = input()
    if filename != "0":
        predict_sample_text(get_vec_from_file(filename))
    else:
        break
