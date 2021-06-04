import tensorflow as tf

device_name = tf.test.gpu_device_name()

import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.utils import np_utils

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
print(set(raw_text))
print(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
print(char_to_int)
print(int_to_char)


class GenCallback(Callback):
    def gen_text(self, size=1000):
        start = np.random.randint(0, n_patterns - 1)
        pattern = dataX[start]
        text = []
        for i in range(size):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = int_to_char[index]
            text.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print("".join(text))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            print(f'{epoch + 1} epochs:')
            self.gen_text()
            print("Seed:")
            print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
print(dataX[0])
print(dataY[0])

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential([
    LSTM(256, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, GenCallback(), TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1)]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list, verbose=2)

filename = "weights-improvement-20.hdf5"
model.load_weights(filename)

start = np.random.randint(0, n_patterns - 1)
pattern = dataX[start]
text = []

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    text.append(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("".join(text))