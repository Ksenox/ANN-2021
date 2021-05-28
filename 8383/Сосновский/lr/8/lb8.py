from matplotlib import gridspec
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.utils import np_utils

class Epochs_CallBack(Callback):
    def gen_text(self, size=100):
        start = np.random.randint(0, n_patterns-1)
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
        if epoch % 2 == 0:
            print(f'{epoch+1} ep:')
            self.gen_text()

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential([
    LSTM(256, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, Epochs_CallBack(), TensorBoard(log_dir='logs', histogram_freq=1, embeddings_freq=1)]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list, verbose=2)

filename = "weights-improvement-20.hdf5"
model.load_weights(filename)

start = np.random.randint(0, n_patterns-1)
pattern = dataX[start]
text = []

for i in range(200):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        text.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
print("".join(text))

