# Load LSTM network and generate text
import sys
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.utils as utils
from tensorflow.keras.callbacks import Callback

# load ascii text and covert to lowercase

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data

n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers

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

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


class TaskCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            start = numpy.random.randint(0, len(dataX) - 1)
            pattern = dataX[start]
            print("Seed:")
            print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
            for i in range(100):
                x = numpy.reshape(pattern, (1, len(pattern), 1))
                x = x / float(n_vocab)
                prediction = model.predict(x, verbose=0)
                index = numpy.argmax(prediction)
                result = int_to_char[index]
                seq_in = [int_to_char[value] for value in pattern]
                sys.stdout.write(result)
                pattern.append(index)
                pattern = pattern[1:len(pattern)]


callbacks_list = [checkpoint, TaskCallback()]
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

