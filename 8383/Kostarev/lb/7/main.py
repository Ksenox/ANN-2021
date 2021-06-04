import numpy as np
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence


# Предсказание ансамблем
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


# Оценка точности ансамбля
def evaluate_ensemble(models, x_data, y_data):
  predictions = get_ensemble_predictions(models, x_data)
  accuracy = predictions == y_data
  return np.count_nonzero(accuracy)/y_data.shape[0]


# Чтение пользовательского текста
def read_txt(filepath, max_words=10000):
    f = open(filepath, 'r')
    txt = f.read().lower()
    txt = re.sub(r"[^a-zA-Z0-9']", " ", txt)  # убираем лишние символы
    txt = txt.split()  # разбиваем на слова
    index = imdb.get_word_index()  # загружаем словарь
    coded = [1]  # индекс начала последовательности
    coded.extend([index.get(i, -1) for i in txt])
    for i in range(len(coded)):
        coded[i] += 3  # смещаем индексы
        if coded[i] >= max_words:
            coded[i] = 2  # отмечаем слова, не вошедшие в число max_words самых популярных
    return coded


# Классификация пользовательского текста
def classify_text(filepath, models, max_review_length, max_words=10000):
  coded_text = sequence.pad_sequences([read_txt(filepath, max_words)], maxlen=max_review_length)
  return get_ensemble_predictions(models, coded_text, False)


# Загрузка и обработка данных
top_words = 10000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=top_words)
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

# Построение модели 1
embedding_vector_length = 32

model_a = Sequential()
model_a.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_a.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_a.add(layers.MaxPooling1D(pool_size=2))
model_a.add(layers.LSTM(100))
model_a.add(layers.Dense(1, activation='sigmoid'))
model_a.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_a.summary())

# Построение модели 2
model_b = Sequential()
model_b.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_b.add(layers.Flatten())
model_b.add(layers.Dense(50, activation = "relu"))
model_b.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model_b.add(layers.Dense(50, activation = "relu"))
model_b.add(layers.Dropout(0.25, noise_shape=None, seed=None))
model_b.add(layers.Dense(50, activation = "relu"))
model_b.add(layers.Dense(1, activation = "sigmoid"))
model_b.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_b.summary())

# Построение модели 3
model_c = Sequential()
model_c.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_c.add(layers.SimpleRNN(32, return_sequences=True))
model_c.add(layers.SimpleRNN(32))
model_c.add(layers.Dense(1, activation = "sigmoid"))
model_c.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_c.summary())

# Построение модели 4
model_d = Sequential()
model_d.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_d.add(layers.GRU(64, recurrent_dropout=0.2))
model_d.add(layers.Dense(1, activation = "sigmoid"))
model_d.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_d.summary())

# Обучение моделей и оценка точности
all_models = [model_a, model_b, model_c, model_d]
k = len(all_models)
train_batch_len = len(Y_train) // k

for i in range(k):
  print("Training model #", i+1);
  train_data = X_train[i*train_batch_len:(i+1)*train_batch_len]
  train_labels = Y_train[i*train_batch_len:(i+1)*train_batch_len]
  all_models[i].fit(train_data, train_labels, validation_data=(X_test, Y_test),
                    epochs=2, batch_size=64)
  scores = all_models[i].evaluate(X_test, Y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1]*100))

print("Ensemble accuracy: %.2f%%" % (evaluate_ensemble(all_models, X_test, Y_test) * 100))

# Обработка пользовательского текста
print(classify_text("test1.txt", all_models, max_review_length, top_words))
print(classify_text("test2.txt", all_models, max_review_length, top_words))
print(classify_text("test3.txt", all_models, max_review_length, top_words))
print(classify_text("test4.txt", all_models, max_review_length, top_words))