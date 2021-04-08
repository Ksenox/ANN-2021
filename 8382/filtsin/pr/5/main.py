from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

train_file = np.genfromtxt('train.csv', delimiter=';')
test_file = np.genfromtxt('test.csv', delimiter=';')

train_data = train_file[:, 0:6]
train_labels = train_file[:, 6]

test_data = test_file[:, 0:6]
test_labels = test_file[:, 6]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


def build_encoder(input):
    encoder = Dense(64, activation='relu')(input)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(4, activation='relu', name='encoder')(encoder)
    return encoder


def build_decoder(input):
    decoder = Dense(32, activation='relu', name='dc1')(input)
    decoder = Dense(64, activation='relu', name='dc2')(decoder)
    decoder = Dense(6, activation='relu', name='decoder')(decoder)
    return decoder


def build_regr(input):
    regr = Dense(64, activation='relu')(input)
    regr = Dense(128, activation='relu')(regr)
    regr = Dense(32, activation='relu')(regr)
    regr = Dense(16, activation='relu')(regr)
    regr = Dense(1, name='regression')(regr)
    return regr


input = Input(shape=(6,))
encoder = build_encoder(input)
decoder = build_decoder(encoder)
regr = build_regr(encoder)

model = Model(
    input,
    outputs=[regr, decoder]
)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=25, validation_split=0.3)
model.evaluate(test_data, [test_labels, test_data])

encoder_model = Model(
    input,
    encoder
)

decoder_model = Model(
    input,
    decoder
)

regr_model = Model(
    input,
    regr
)

encoder_model.save('encoder.h5')
decoder_model.save('decoder.h5')
regr_model.save('regr.h5')

encode_data = encoder_model.predict(test_data)
decode_data = decoder_model.predict(test_data)
regr_data = regr_model.predict(test_data)

decode_data = decode_data * std + mean

print(regr_data)
print(test_labels)

#

np.savetxt('encode.csv', encode_data, delimiter=';')
np.savetxt('decode.csv', decode_data, delimiter=';')
np.savetxt('regr.csv', np.vstack((regr_data[:, 0], test_labels)).transpose(), delimiter=';')
