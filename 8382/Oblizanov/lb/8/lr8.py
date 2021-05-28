import datetime
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.callbacks
from keras.utils import np_utils
import sys

input_file = "wonderland.txt"
output_file = "generated.txt"
raw_text = open(input_file).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
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
X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)


class MyCustomCallback(keras.callbacks.Callback):

    def __init__(self):
        super(MyCustomCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 4 == 0:
            generate(self.model, epoch + 1)


def generate(model, epoch):
    g = open(output_file, 'a')
    gen_symbols = []
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    g.write(
        "Epoch: " + str(epoch) + ". Seed:\n" + "\"" + ''.join([int_to_char[value] for value in pattern]) + "\"\n")
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        gen_symbols.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    g.write("generated text is:\n" + "\"" + ''.join(gen_symbols) + "\"\n")
    g.close()


def build_model():
    model = Sequential()
    model.add(LSTM(400, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def run_fit():
    model = build_model()
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, MyCustomCallback(), tensorboard_callback]
    model.fit(X, y, epochs=30, batch_size=64, callbacks=callbacks_list)


    gen_symbols = []
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        gen_symbols.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("generated text is:\n" + "\"" + ''.join(gen_symbols) + "\"\n")


run_fit()
