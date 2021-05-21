import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence


def get_ensemble_predictions(models, x_data):
    return np.mean(np.asarray([model.predict(x_data) for model in models]), 0).flatten()


def evaluate_ensemble(models, x_data, y_data):
    predictions = np.round(get_ensemble_predictions(models, x_data))
    accuracy = predictions == y_data
    return np.count_nonzero(accuracy) / y_data.shape[0]


def sequence_from_file(filepath, max_words=10000):
    with open(filepath, "r") as f:
        content = f.read().lower()
        words = [word.strip(":;?!.,'\"-_").lower() for word in content.strip().split()]
        index = imdb.get_word_index()
        return list(map(lambda x: 2 if x >= max_words else x + 3, [1] + [index.get(word, -1) for word in words]))


def classify_text(filepath, models, max_review_length, max_words=10000):
    coded_text = sequence.pad_sequences([sequence_from_file(filepath, max_words)], maxlen=max_review_length)
    prediction = get_ensemble_predictions(models, coded_text)
    print(f"Result for the text {filepath} is {prediction}")


def build_simple_lstm_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model_with_conv_pool():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_lstm_dense_model():
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


top_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
X_train = data[:40000]
y_train = targets[:40000]
X_test = data[40000:]
y_test = targets[40000:]

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
models = [build_simple_lstm_model(), build_model_with_conv_pool(), build_lstm_dense_model()]

for index, model in enumerate(models):
    print(f"Training model #{index+1}")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=2, batch_size=64)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

print("Ensemble accuracy: %.2f%%" % (evaluate_ensemble(models, X_test, y_test) * 100))

classify_text("reviews/good1.txt", models, 500)
