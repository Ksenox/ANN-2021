import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

np.random.seed(42)

train_data = np.genfromtxt("samples/train.csv", delimiter=",")
test_data = np.genfromtxt("samples/test.csv", delimiter=",")

train_x, train_y = train_data[:, :6], train_data[:, 6]
test_x, test_y = test_data[:, :6], test_data[:, 6]

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

basic_input_layer = Input(shape=(6,))


def create_encoder() -> Dense:
    encoded = Dense(64, activation='relu')(basic_input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(8, activation='relu', name='encode')(encoded)
    return encoded


def create_decoder(input_layer: Dense) -> Dense:
    decoded = Dense(32, activation='relu')(input_layer)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(6, name='decode')(decoded)
    return decoded


def create_regression(input_layer: Dense) -> Dense:
    predicted = Dense(64, activation='relu')(input_layer)
    predicted = Dense(32, activation='relu')(predicted)
    predicted = Dense(16, activation='relu')(predicted)
    predicted = Dense(8, activation='relu')(predicted)
    predicted = Dense(1, name="predict")(predicted)
    return predicted


encoder_output = create_encoder()
decoder_output = create_decoder(encoder_output)
regression_output = create_regression(encoder_output)

encoder = Model(basic_input_layer, encoder_output, name='encoder')
decoder = Model(basic_input_layer, decoder_output, name='decoder')
regression = Model(basic_input_layer, regression_output, name='regression')

model = Model(inputs=[basic_input_layer], outputs=[
    decoder_output,
    regression_output
])

model.compile(optimizer='adam', loss='mse', metrics='mae')
history = model.fit(train_x, [train_x, train_y], epochs=100, batch_size=10,
                    verbose=1, validation_split=0.8)

encoded_test = encoder.predict(test_x)
decoded_test = decoder.predict(test_x)
regression_test = regression.predict(test_x)

decoded_test = decoded_test * std + mean

np.savetxt('samples/encoded.csv', encoded_test, delimiter=',', fmt='%1.3f')
np.savetxt('samples/decoded.csv', decoded_test, delimiter=',', fmt='%1.3f')
np.savetxt('samples/regression.csv', np.vstack((test_y, regression_test[:, 0])).T, delimiter=',', fmt='%1.3f')

encoder.save('models/encoder.h5')
decoder.save('models/decoder.h5')
regression.save('models/regression.h5')

