import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence

max_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=max_words)
train_x = np.concatenate((training_data, testing_data), axis=0)
train_y = np.concatenate((training_targets, testing_targets), axis=0)
train_y = np.array(train_y).astype("float32")

max_review_length = 500
train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
embedding_vector_length = 32

# ---------------- 1 MODEL ----------------

# model = Sequential()
# model.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))

# ---------------- 2 MODEL ----------------

# model = Sequential()
# model.add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))

# ---------------- 3 MODEL ----------------

models = []
scores = []
m = 4
for i in range(m):
    x_train = train_x[int(len(train_x) / m) * i: int(len(train_x) / m) * (i + 1)]
    y_train = train_y[int(len(train_y) / m) * i: int(len(train_y) / m) * (i + 1)]

    models.append(Sequential())
    models[i].add(Embedding(max_words, embedding_vector_length, input_length=max_review_length))
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

print("Mean accuracy: %.2f%%" % (np.mean(scores) * 100))


def get_vec_from_file(file):
    text = ""
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
    print("Good" if prediction == 1 else "Bad")


predict_sample_text(get_vec_from_file("sample1"))
predict_sample_text(get_vec_from_file("sample2"))
predict_sample_text(get_vec_from_file("sample3"))
predict_sample_text(get_vec_from_file("sample4"))
predict_sample_text(get_vec_from_file("sample5"))
predict_sample_text(get_vec_from_file("sample6"))
