import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import Callback, TensorBoard


def generator(model, dataX, int_to_char):
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]

    print("###---### Seed:")
    print(''.join([int_to_char[value] for value in pattern]))

    result = ''
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result += int_to_char[index]
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("###---### Generated:")
    print(result)
    print("###---### Done")


class TextGenerator(Callback):
    def __init__(self, interval, generator, dataX, int_to_char):
        super(TextGenerator, self).__init__()
        self.interval = interval
        self.generator = generator
        self.dataX = dataX
        self.int_to_char = int_to_char

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 or epoch == self.params["epochs"] - 1:
            self.generator(self.model, self.dataX, self.int_to_char)


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

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

file_path = 'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
text_generator = TextGenerator(4, generator, dataX, int_to_char)
tensorboard = TensorBoard(histogram_freq=1)
callbacks_list = [checkpoint, text_generator, tensorboard]

model.fit(X, Y, epochs=30, batch_size=128, callbacks=callbacks_list)


filename = 'weights-improvement-30-1.7816.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

generator(model, dataX, int_to_char)
