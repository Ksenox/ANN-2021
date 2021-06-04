import numpy as np
import re
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
import pandas
from sklearn.utils import shuffle


# Callback, делающий предсказания для случайного твита в после каждой эпохи
class RandomPredictionCb(keras.callbacks.Callback):
    def __init__(self, interval, data, labels, tokenizer):
        super(RandomPredictionCb, self).__init__()
        self.interval = interval
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:
            rn = np.random.randint(0, len(self.data) - 1)
            prediction = self.model.predict(np.asarray([self.data[rn]]))
            txt = self.tokenizer.sequences_to_texts([self.data[rn]])
            print("Text: ", txt)
            print("Prediction: ", prediction)
            print("Real value: ", self.labels[rn].flatten())


# Вычисление предсказаний ансамбля
def get_ensemble_predictions(models, sequence, round=True):
    predictions = []
    for model in models:
        curr_prediction = model.predict(sequence)
        predictions.append(curr_prediction)
    predictions = np.asarray(predictions)
    predictions = np.mean(predictions, 0)
    if round:
        predictions = np.round(predictions)
    return predictions.flatten()


# Оценка точности предсказаний ансамбля
def evaluate_ensemble(models, x_data, y_data):
    predictions = get_ensemble_predictions(models, x_data)
    accuracy = predictions == y_data
    return np.count_nonzero(accuracy)/y_data.shape[0]


# Генератор модели 1-го типа
def get_model_type_one(top_words, length):
    embedding_vector_length = 32
    model_one = Sequential()
    model_one.add(layers.Embedding(top_words, embedding_vector_length, input_length=length))
    model_one.add(layers.Flatten())
    model_one.add(layers.Dense(200, activation='relu'))
    model_one.add(layers.Dropout(0.3))
    model_one.add(layers.Dense(200, activation='relu'))
    model_one.add(layers.Dropout(0.4))
    model_one.add(layers.Dense(1, activation='sigmoid'))
    model_one.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_one


# Генератор модели 2-го типа
def get_model_type_two(num, length):
    embedding_vector_length = 32
    model_two = Sequential()
    model_two.add(layers.Embedding(num, embedding_vector_length, input_length=length))
    model_two.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model_two.add(layers.MaxPooling1D(pool_size=2))
    model_two.add(layers.Dropout(0.2))
    model_two.add(layers.LSTM(200, return_sequences=True))
    model_two.add(layers.Dropout(0.2))
    model_two.add(layers.LSTM(200))
    model_two.add(layers.Dropout(0.2))
    model_two.add(layers.Dense(1, activation='sigmoid'))
    model_two.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_two


# Генератор модели 3-го типа
def get_model_type_three(num, length):
    embedding_vector_length = 32
    model_three = Sequential()
    model_three.add(layers.Embedding(num, embedding_vector_length, input_length=length))
    model_three.add(layers.LSTM(300, return_sequences=True))
    model_three.add(layers.Dropout(0.2))
    model_three.add(layers.LSTM(300))
    model_three.add(layers.Dropout(0.3))
    model_three.add(layers.Dense(1, activation='sigmoid'))
    model_three.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_three


# Обработка текста твитов
def process_data(series):
    pat = r"@.[^\s]+|https?:\/\/.[^\s]+"
    data = series.str.lower()
    data = data.str.replace(pat, "")
    data = data.str.replace(r":\)|: \)|:-\)|;-\)|:d|:p|;\)|;d|;p|=\)", " yppahelims ")
    data = data.str.replace(r":\(|: \(|:-\(", " daselims ")
    data = data.str.replace(r":0|:o", " esirpruselims ")
    data = data.str.replace(r":@", " taelims ")
    return data


# Обработка пользовательского текста
def process_string(string: str):
    pat = r"@.[^\s]+|https?:\/\/.[^\s]+"
    string = string.lower()
    string = re.sub(pat, "", string)
    string = re.sub(r":\)|: \)|:-\)|;-\)|:d|:p|;\)|;d|;p|=\)", " yppahelims ", string)
    string = re.sub(r":\(|: \(|:-\(", " daselims ", string)
    string = re.sub(r":0|:o", " esirpruselims ", string)
    string = re.sub(r":@", " taelims ", string)
    return string


# Функция для классификации пользовательского текста
def dialog(models, tokenizer, max_length):
    while True:
        data = input('Your text: ')
        if data == 'exit':
            break
        data = process_data(pandas.Series([data]))
        data = tokenizer.texts_to_sequences(data)
        data = np.asarray(data)
        data = sequence.pad_sequences(data, maxlen=max_length)
        result = get_ensemble_predictions(models, np.asarray(data), round=False)
        print('Prediction: ', result)
        print('\n')


# Загрузка датасета
train_path = "training.csv"
train_df = pandas.read_csv(train_path, header=None, encoding='latin-1')

# Обработка датасета
train_df = shuffle(train_df)
train_data = process_data(train_df[5])
train_data = np.asarray(train_data)
train_labels = np.asarray(train_df[0])

# Кодирование твитов
max_words = 30000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(train_data)
train_data = tokenizer.texts_to_sequences(train_data)

max_length = 50
train_data = sequence.pad_sequences(train_data, maxlen=max_length)

# Нормализация выходных данных
train_labels = train_labels/4

# Разбиение выборки на тренировочную и тестовую
test_num = train_data.shape[0] // 10
test_data = train_data[:test_num]
test_labels = train_labels[:test_num]
train_data = train_data[test_num:]
train_labels = train_labels[test_num:]


print("Here we go again")

# Построение моделей
all_models = [
    get_model_type_one(max_words, max_length),
    get_model_type_two(max_words, max_length),
    get_model_type_three(max_words, max_length)
]
k = len(all_models)
train_batch_len = len(train_labels) // k

# Обучение моделей и оценка их точности
for i in range(k):
    train_data_k = train_data[i*train_batch_len:(i+1)*train_batch_len]
    train_labels_k = train_labels[i*train_batch_len:(i+1)*train_batch_len]

    earlystopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                     patience=1, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=f'logs/{i}', histogram_freq=1)
    randomprediction_cb = RandomPredictionCb(1, test_data, test_labels, tokenizer)
    callback_list = [earlystopping_cb, tensorboard_cb, randomprediction_cb]

    all_models[i].fit(train_data_k, train_labels_k, validation_split=0.1,
                      epochs=10, batch_size=128, callbacks=callback_list)
    scores = all_models[i].evaluate(test_data, test_labels, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print("\n")

# Вывод точности ансамбля
print("Ensemble accuracy: %.2f%%" % (evaluate_ensemble(all_models, test_data, test_labels) * 100))

# Тестирование модели на пользовательском тексте
dialog(all_models, tokenizer, max_length)