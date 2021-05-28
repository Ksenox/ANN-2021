import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils


def generate(model, dataX, dictionary):
    gen_symbols = []
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:\n" + "\"" + ''.join([dictionary[value] for value in pattern]) + "\"\n")
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = dictionary[index]
        gen_symbols.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("Generated text:\n" + "\"" + ''.join(gen_symbols) + "\"\n")


class CustomCallBack(keras.callbacks.Callback):
    def __init__(self, interval, generator, dataX, dictionary):
        super(CustomCallBack, self).__init__()
        self.interval = interval
        self.generator = generator
        self.dataX = dataX
        self.dictionary = dictionary

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
          self.generator(self.model, self.dataX, self.dictionary)


def build_model():
    model = Sequential()
    model.add(LSTM(300, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(300))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model



filename = "wonderland.txt"
f = open(filename)
raw_text = f.read()
raw_text = raw_text.lower()
f.close()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 150
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
y = utils.to_categorical(dataY)

model = build_model()


filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
customCB = CustomCallBack(10, generate, dataX, int_to_char)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, customCB, tensorboard_callback]


model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)

#filename = "weights.hdf5"
#model.load_weights(filename)
#model.compile(loss='categorical_crossentropy', optimizer='adam')

generate(model, dataX, int_to_char)


