import re
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.utils import np_utils


class GeneratorLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        text = gen_text(self.model)
        with open("/content/drive/MyDrive/Colab Notebooks/8/texts/text_" + str(epoch) + ".txt", "w") as file:
            file.write(text)


def gen_text(model):
  start = numpy.random.randint(0, len(dataX)-1)
  pattern = dataX[start]
  result = []
  for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result.append(int_to_char[index])
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
  return "".join(result)


filename = "/content/drive/MyDrive/Colab Notebooks/8/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower().replace("*", "")
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

filepath = "/content/drive/MyDrive/Colab Notebooks/8/weights/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='/content/drive/MyDrive/Colab Notebooks/8/tensorboard', histogram_freq=1,
                          embeddings_freq=1),
callbacks_list = [checkpoint, tensorboard, GeneratorLogger()]

model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)
