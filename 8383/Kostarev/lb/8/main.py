import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.utils import np_utils


def generator(model, dataX, dictionary):
    start = numpy.random.randint(0, len(dataX) - 1)
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


class TextGeneratorCallback(Callback):
    def __init__(self, interval, generator, dataX, dictionary):
        super(TextGeneratorCallback, self).__init__()
        self.interval = interval
        self.generator = generator
        self.dataX = dataX
        self.dictionary = dictionary

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:
            self.generator(self.model, self.dataX, self.dictionary)


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

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
textgenerator = TextGeneratorCallback(3, generator, dataX, int_to_char)
tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, textgenerator, tensorboard]

model.fit(X, y, epochs=10, batch_size=256, callbacks=callbacks_list)

filename = "weights-improvement-10-2.4127.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

generator(model, dataX, int_to_char)