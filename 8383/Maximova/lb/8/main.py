import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard

def lookupTable(n_chars, seq_length, raw_text, chars_to_int):
    dataX = []
    dataY = []

    for i in range(0, n_chars - seq_length, 1): # 144408 - 100: start
        stop = i + seq_length
        seq_in = raw_text[i : stop] # 0 - 100; 1 - 101; ...;
        seq_out = raw_text[stop]

        dataX.append([chars_to_int[char] for char in seq_in])
        dataY.append([chars_to_int[seq_out]])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    return dataX, dataY, n_patterns

def dataPreparation(dataX, dataY, n_patterns, seq_length, n_vocab):
    # 1) преобр. Х в [образцы, временные шаги, особенности]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))  # 144308 шаблонов длины 100 столбцами

    # изменение масштаба от 0 - 1 для облегчения изучения шаблонов сетью
    X = X / float(n_vocab)

    # one hot encoded Y
    Y = np_utils.to_categorical(dataY)
    return X, Y

def buildModel(X, Y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    return model

def generateTextLSTM(dataX, dataY, model, n_vocab, size=2500):
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    for i in range(size):
        X = np.reshape(pattern, (1, len(pattern), 1))  # 1 шаблон длины len(pattern) в столбец
        X = X / float(n_vocab)

        prediction = model.predict(X, verbose=0)
        index = np.argmax(prediction)  # argmax - возвращает индекс максимального значения вдоль указанной оси

        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)

        pattern.append(index)
        pattern = pattern[1 : len(pattern)]
    print('\nDone.')

class CallBackGT(Callback):
    def __init__(self, numb_epoch):
        self.numb_epoch = numb_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % int(self.numb_epoch) == 0:
            generateTextLSTM(dataX, dataY, self.model, n_vocab)


print("Этап 1: Обучение модели для поиска оптимальных весов")
print("Этап 2: Генерация текста нейронной сетью")
stage = input("Введите номер этапа ")

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# print(raw_text)

# set - содержит неповторяющиеся элементы
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# print(chars_to_int)

n_chars = len(raw_text)
n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Total Vocab: ", n_vocab)
seq_length = 100
dataX, dataY, n_patterns = lookupTable(n_chars, seq_length, raw_text, chars_to_int)

if stage == "1":
    numb_epoch = input("Введите частоты генерации текста при обучении ")
    X, Y = dataPreparation(dataX, dataY, n_patterns, seq_length, n_vocab)
    model = buildModel(X, Y)

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # для сохранения весов лучшей модели (в данном случае сохраняется вся модель)
    # в файле в конце каждой эпохи (по умолчанию)
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',  # отслеживаемая метрика - потери
        verbose=1,  # режим детализации
        save_best_only=True,  # сохранение для лучшей модели
        mode='min')  # решение о перезаписи в случае минимальных потерь

    callbacks_list = [
        checkpoint,
        CallBackGT(numb_epoch),
        TensorBoard(
            log_dir="logs",
            # путь к каталогу, в котором сохраняются файлы журнала для анализа TensorBoard
            histogram_freq=1,
            # частота (в эпохах), с которой вычисляются гистрограммы активации и веса для слоев модели
            embeddings_freq=1
            # частота (в эпохах), с которой будут визуализироваться встраиваемые слои

        )]

    model.fit(
        X, Y,
        epochs=30,
        batch_size=64,
        callbacks=callbacks_list)

elif stage == "2":
    model = load_model("weights-improvement-30-1.6691.hdf5")
    generateTextLSTM(dataX, dataY, model, n_vocab)



