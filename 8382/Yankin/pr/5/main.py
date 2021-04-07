from keras import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np


train = np.genfromtxt("train.csv", delimiter=";")
train_data = np.reshape(train[:, 0:6], (len(train), 6))
train_labels = np.reshape(train[:, 6], (len(train), 1))

test = np.genfromtxt("test.csv", delimiter=";")
test_data = np.reshape(test[:, 0:6], (len(test), 6))
test_labels = np.reshape(test[:, 6], (len(test), 1))


encoded_size = 4

main_input = Input(shape=(6,), name='main_input')

layer_encoder = Dense(36, activation='relu')(main_input)
layer_encoder = Dense(36, activation='relu')(layer_encoder)
layer_encoder = Dense(encoded_size, activation='relu', name='encoder_output')(layer_encoder)

layer_decoder = Dense(36, activation='relu')(layer_encoder)
layer_decoder = Dense(36, activation='relu')(layer_decoder)
layer_decoder = Dense(6, name='decoder_output')(layer_decoder)

layer_regression = Dense(18, activation='relu')(layer_encoder)
layer_regression = Dense(36, activation='relu')(layer_regression)
layer_regression = Dense(36, activation='relu')(layer_regression)
layer_regression = Dense(18, activation='relu')(layer_regression)
layer_regression = Dense(1, name='regression_output')(layer_regression)


model = Model(inputs=[main_input], outputs=[layer_regression, layer_decoder])
model.compile(optimizer='adam', loss='mse', metrics='mae')
model.fit(train_data, [train_labels, train_data], epochs=200, batch_size=25)

model.evaluate(test_data, [test_labels, test_data])

encoder_model = Model(main_input, layer_encoder)
decoder_model = Model(main_input, layer_decoder)
regression_model = Model(main_input, layer_regression)

encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
regression_model.save('regression_model.h5')

np.savetxt('encoded.csv', encoder_model.predict(test_data), delimiter=';')
np.savetxt('decoded.csv', decoder_model.predict(test_data), delimiter=';')
np.savetxt('regression.csv', np.hstack((test_labels, regression_model.predict(test_data))), delimiter=';')
