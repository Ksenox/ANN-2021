# %load_ext tensorboard

import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils


# Генерация текста
def generate_text(model, dataX, dictionary):
  start = numpy.random.randint(0, len(dataX)-1)
  pattern = dataX[start]
  print("Seed:")
  print("\"", ''.join([dictionary[value] for value in pattern]), "\"\n")
  n_vocab = len(dictionary)
  text = ''.join([dictionary[value] for value in pattern])

  for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = dictionary[index]
    seq_in = [dictionary[value] for value in pattern]
    text = text + result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
  print(text)


# Callback
class TextGenCallback(keras.callbacks.Callback):
    def __init__(self, interval, generator, dataX, dictionary):
        super(TextGenCallback, self).__init__()
        self.interval = interval
        self.generator = generator
        self.dataX = dataX
        self.dictionary = dictionary
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:
          self.generator(self.model, self.dataX, self.dictionary)


filename = "wonderland.txt"
f = open(filename)
raw_text = f.read()
raw_text = raw_text.lower()
f.close()

# Составление словаря
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

# Составление последовательностей
seq_length = 200
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# Подготовка данных для обучения
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = utils.to_categorical(dataY)

# Построение модели
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Добавление CallBack'ов
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
text_gen = TextGenCallback(10, generate_text, dataX, int_to_char)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, text_gen, tensorboard_callback]

# Удаление предыдущих логов
!rm -rf ./logs/

# Обучение модели
model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)

# %tensorboard --logdir logs/

# Загрузка наилучших весов
filename = "weights-improvement-30-1.5393.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Генерация текста
generate_text(model, dataX, int_to_char)
