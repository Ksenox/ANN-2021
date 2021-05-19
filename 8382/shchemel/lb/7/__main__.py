from typing import Dict, List, Tuple

import numpy as np
from keras.datasets import imdb
from keras.layers import GRU, LSTM, Conv1D, Dense, MaxPooling1D, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.preprocessing import sequence

INDEX = imdb.get_word_index()


def load_data(
    num_words=5000, max_review_length=500
) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(
        num_words=num_words
    )
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    train_length = (data.shape[0] // 10) * 8
    train_data = data[:train_length]
    test_data = data[train_length:]
    train_targets = targets[:train_length]
    test_targets = targets[train_length:]

    train_data = sequence.pad_sequences(train_data, maxlen=max_review_length)
    test_data = sequence.pad_sequences(test_data, maxlen=max_review_length)
    return (train_data, train_targets), (test_data, test_targets)


def convert_text(text: str, max_size=5000) -> np.array:
    result = []

    for word in text.split():
        index = INDEX.get(word.lower())
        if index is None or index + 3 > max_size:
            continue
        result.append(index + 3)

    reverse_index = dict([(value, key) for (key, value) in INDEX.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in result])
    print(decoded)

    return result


def ensemble_predictions(models: List[Model], data: np.array) -> float:
    predictions = [model.predict(data) for model in models]
    return np.mean(np.asarray(predictions), 0).flatten()


def evaluate_ensemble(models: List[Model], data: np.array, targets: np.array) -> float:
    predictions = np.round(ensemble_predictions(models, data))
    return np.count_nonzero(predictions == targets) / targets.shape[0]


def create_lstm_model(top_words=5000, embedding_vecor_length=32, max_review_length=500) -> Model:
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_simple_rnn_model(
    top_words=5000, embedding_vecor_length=32, max_review_length=500
) -> Model:
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(SimpleRNN(64, return_sequences=True, dropout=0.2))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def create_gru_model(top_words=5000, embedding_vecor_length=32, max_review_length=500) -> Model:
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True, dropout=0.2))
    model.add(GRU(16))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def main(review: str):
    (train_data, train_targets), (test_data, test_targets) = load_data()
    models = [create_lstm_model(), create_simple_rnn_model(), create_gru_model()]
    for index, model in enumerate(models):
        print(f"Model {index + 1}")
        print(model.summary())
        model.fit(
            train_data,
            train_targets,
            validation_data=(test_data, test_targets),
            epochs=3,
            batch_size=64,
        )
        scores = model.evaluate(test_data, test_targets)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    print(f"Ensemble accuracy: {evaluate_ensemble(models, test_data, test_targets)}")

    review_vector = convert_text(review)
    review_vector = sequence.pad_sequences([review_vector], maxlen=500)
    prediction = ensemble_predictions(models, review_vector)
    print(f"Prediction for review is: {prediction}")


if __name__ == "__main__":
    text = input()
    main(text)
