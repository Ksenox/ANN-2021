import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb


top_words = 20000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
targets = np.array(targets).astype("float32")
# test_x = data[:10000]
# test_y = targets[:10000]
# train_x = data[10000:]
# train_y = targets[10000:]

max_review_length = 500
# train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
# test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)
embedding_vector_length = 128

for i in range(0, 5):
    test_x = data[i * 10000:(i + 1) * 10000]
    test_y = targets[i * 10000:(i + 1) * 10000]
    train_x = np.concatenate([data[:i * 10000], data[(i + 1) * 10000:]], axis=0)
    train_y = np.concatenate([targets[:i * 10000], targets[(i + 1) * 10000:]], axis=0)

    train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)
    test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)

    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=64)
    model.save('model' + str(i) + '.h5')
    # models.append(model)

    # scores = model.evaluate(test_x, test_y, verbose=False)
    # print("Accuracy: %.2f%%" % (scores[1]*100))


