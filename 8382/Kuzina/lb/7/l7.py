import numpy as np
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence


def get_ensemble_predictions(models, sequence, round = True):
  predictions = []
  for model in models:
    curr_prediction = model.predict(sequence)
    predictions.append(curr_prediction)
  predictions = np.asarray(predictions)
  predictions = np.mean(predictions, 0)
  if round:
    predictions = np.round(predictions)
  return predictions.flatten()


def evaluate_ensemble(models, x_data, y_data):
  predictions = get_ensemble_predictions(models, x_data)
  accuracy = predictions == y_data
  return np.count_nonzero(accuracy)/y_data.shape[0]


def read_txt(filepath, max_words=5000):
    f = open(filepath, 'r')
    txt = f.read().lower()
    txt = re.sub(r"[^a-z0-9']", " ", txt)
    txt = txt.split()
    index = imdb.get_word_index()
    coded = [-2]
    coded.extend([index.get(i, -1) for i in txt])
    for i in range(len(coded)):
        coded[i] += 3
        if coded[i] >= max_words:
            coded[i] = 2
    return coded


def classify_text(filepath, models, max_review_length, max_words=5000):
  coded_text = sequence.pad_sequences([read_txt(filepath, max_words)], maxlen=max_review_length)
  return get_ensemble_predictions(models, coded_text, False)


def get_model_type_one(num_, max_review_length_ ):
    embedding_vector_length = 32
    model_one = Sequential()
    model_one.add(layers.Embedding(num_, embedding_vector_length, input_length=max_review_length_))
    model_one.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_one.add(layers.MaxPooling1D(pool_size=2))
    model_one.add(layers.LSTM(100))
    model_one.add(layers.Dense(1, activation='sigmoid'))
    model_one.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_one


def get_model_type_two(num_, max_review_length_ ):
    embedding_vector_length = 32
    model_two = Sequential()
    model_two.add(layers.Embedding(num_, embedding_vector_length, input_length=max_review_length_))
    model_two.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_two.add(layers.MaxPooling1D(pool_size=2))
    model_two.add(layers.Dropout(0.2))
    model_two.add(layers.LSTM(100))
    model_two.add(layers.Dropout(0.2))
    model_two.add(layers.Dense(1, activation='sigmoid'))
    model_two.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model_two


num = 5000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num)
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


all_models = [get_model_type_one(num, max_review_length), get_model_type_one(num, max_review_length), get_model_type_two(num, max_review_length), get_model_type_two(num, max_review_length)]
k = len(all_models)
train_batch_len = len(Y_train) // k

for i in range(k):
  train_data = X_train[i*train_batch_len:(i+1)*train_batch_len]
  train_labels = Y_train[i*train_batch_len:(i+1)*train_batch_len]
  all_models[i].fit(train_data, train_labels, validation_data=(X_test, Y_test),epochs=2, batch_size=64)
  scores = all_models[i].evaluate(X_test, Y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1]*100))
  print("\n")

print("Ensemble accuracy: %.2f%%" % (evaluate_ensemble(all_models, X_test, Y_test) * 100))


print(classify_text("1.txt", all_models, max_review_length, num))
print(classify_text("2.txt", all_models, max_review_length, num))
print(classify_text("3.txt", all_models, max_review_length, num))
print(classify_text("4.txt", all_models, max_review_length, num))
