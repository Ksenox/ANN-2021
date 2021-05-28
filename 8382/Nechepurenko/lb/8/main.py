import numpy
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


def generate_text(model, dataX, dictionary):
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


class GenerateText(Callback):
    def __init__(self, each_nth_iteration, generator_fn, dataX, int_to_char_map):
        super().__init__()
        self.each_nth_iteration = each_nth_iteration
        self.generator_fn = generator_fn
        self.dataX = dataX
        self.int_to_char_map = int_to_char_map

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.each_nth_iteration == 0:
            self.generator_fn(self.model, self.dataX, self.int_to_char_map)


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

seq_length = 200
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

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
text_gen = GenerateText(10, generate_text, dataX, int_to_char)
tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, text_gen, tensorboard_callback]
model.fit(X, y, epochs=60, batch_size=128, callbacks=callbacks_list)

# filename = "weights-improvement-59-1.2606.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')


# %load_ext tensorboard
# %tensorboard --logdir logs/
