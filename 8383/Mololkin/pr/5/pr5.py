import os
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model


def generate(length):
    data = np.zeros([length, 6])
    x = np.random.normal(0, 10, length)
    e = np.random.normal(0, 0.3, length)

    data[:, 0] = np.array(x ** 2 + x + e)
    data[:, 1] = np.array(np.fabs(x) + e)
    data[:, 2] = np.array(np.sin(x - np.pi / 4) + e)
    data[:, 3] = np.array(np.log(np.fabs(x)) + e)
    data[:, 4] = np.array(-(x ** 3) + e)
    data[:, 5] = np.array(-x + e)

    labels = -x / 4 + e

    return data, labels


train_data, train_labels = generate(1000)
test_data, test_labels = generate(100)

os.mkdir("./input_data")
np.savetxt("./input_data/train_data.csv", train_data, delimiter=";")
np.savetxt("./input_data/train_labels.csv", train_labels, delimiter=";")
np.savetxt("./input_data/test_data.csv", test_data, delimiter=";")
np.savetxt("./input_data/test_labels.csv", test_labels, delimiter=";")

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

_input = Input(shape=(6, ))

encoder = Dense(64, activation='relu')(_input)
encoder = Dense(16, activation='relu')(encoder)
encoder = Dense(4, activation='relu')(encoder)

decoder = Dense(16, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dense(6)(decoder)

regress = Dense(16, activation='relu')(encoder)
regress = Dense(64, activation='relu')(regress)
regress = Dense(32, activation='relu')(regress)
regress = Dense(1)(regress)

model = Model(inputs=_input, outputs=[decoder, regress])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_data, [train_data, train_labels], epochs=100, batch_size=10)

encoder_m = Model(inputs=_input, outputs=decoder)
decoder_m = Model(inputs=_input, outputs=encoder)
regress_m = Model(inputs=_input, outputs=regress)

encoder_res = encoder_m.predict(test_data)
decoder_res = decoder_m.predict(test_data)
regression_res = regress_m.predict(test_data)

os.mkdir("./result")
np.savetxt("./result/encoded.csv", encoder_res, delimiter=";")
np.savetxt("./result/decoded.csv", decoder_res, delimiter=";")
np.savetxt("./result/regression.csv", np.array([regression_res[:, 0], test_labels]).transpose(), delimiter=";")

model.save('./models/encoder_m.h5')
model.save('./models/decoder_m.h5')
model.save('./models/regression_m.h5')