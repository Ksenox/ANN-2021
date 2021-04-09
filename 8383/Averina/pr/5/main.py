from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

train_file = np.genfromtxt('train_data.csv', delimiter=' ')
test_file = np.genfromtxt('test_data.csv', delimiter=' ')

train_data = train_file[:, 0:6]
train_labels = train_file[:, 6]

test_data = test_file[:, 0:6]
test_labels = test_file[:, 6]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


def func_encoder(input):
    e = Dense(64, activation='relu')(input)
    e = Dense(32, activation='relu')(e)
    e = Dense(4, activation='relu', name='encoder')(e)
    return e


def func_decoder(input):
    d = Dense(32, activation='relu', name='dc1')(input)
    d = Dense(64, activation='relu', name='dc2')(d)
    d = Dense(6, activation='relu', name='decoder')(d)
    return d


def func_regr(input):
    r = Dense(64, activation='relu')(input)
    r = Dense(128, activation='relu')(r)
    r = Dense(32, activation='relu')(r)
    r = Dense(16, activation='relu')(r)
    r = Dense(1, name='regression')(r)
    return r


input = Input(shape=(6,))
encoder = func_encoder(input)
decoder = func_decoder(encoder)
regression = func_regr(encoder)

model = Model(input, outputs=[regression, decoder])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=25, validation_split=0.3)
model.evaluate(test_data, [test_labels, test_data])

encoder_model = Model(input, encoder)
decoder_model = Model(input, decoder)
regression_model = Model(input, regression)

encoder_model.save('encoder.h5')
decoder_model.save('decoder.h5')
regression_model.save('regression.h5')

encode_data = encoder_model.predict(test_data)
decode_data = decoder_model.predict(test_data)
regression_data = regression_model.predict(test_data)

decode_data = decode_data * std + mean

print(regression_data[:4])
print(test_labels[:4])

np.savetxt('encode.csv', encode_data, delimiter=';')
np.savetxt('decode.csv', decode_data, delimiter=';')
np.savetxt('regression.csv', np.vstack((regression_data[:, 0], test_labels)).transpose(), delimiter=';')
