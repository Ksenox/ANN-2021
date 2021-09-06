# -*- coding: utf-8 -*-
"""INS_LAB8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1egAW9GZttysIS5bV-v2MoxG6m6qeCJC3
"""

import re
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.utils import np_utils
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class MyCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
    text = gen_text(self.model)
    with open(f"{COLAB_PREFIX}\gererated_text\{epoch}.txt", 'w') as file:
      file.write(text)

# pick a random seed
def gen_text(model):
  start = numpy.random.randint(0, len(dataX)-1)
  pattern = dataX[start]
  result = []
  print("Seed:")
  print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
  # generate characters
  for i in range(1000):
          x = numpy.reshape(pattern, (1, len(pattern), 1))
          x = x / float(n_vocab)
          prediction = model.predict(x, verbose=0)
          index = numpy.argmax(prediction)
          result.append(int_to_char[index])
          pattern.append(index)
          pattern = pattern[1:len(pattern)]
  return "".join(result)

COLAB_PREFIX = "/content/sample_data"
filename = "/content/drive/MyDrive/Colab Notebooks/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = raw_text.replace("*", "")
raw_text = re.sub(" +", " ", raw_text)
raw_text = re.sub("\n+", "\n", raw_text)

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

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath= COLAB_PREFIX + "/weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=f"/{COLAB_PREFIX}/tensorboard", histogram_freq=1,
                          embeddings_freq=1),
callbacks_list = [checkpoint, tensorboard, MyCallback()]

model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)