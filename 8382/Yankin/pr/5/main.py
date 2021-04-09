from keras import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np


train = np.genfromtxt("train.csv", delimiter=";")
train_data = np.reshape(train[:, 0:6], (len(train), 6))
train_labels = np.reshape(train[:, 6], (len(train), 1))

test = np.genfromtxt("test.csv", delimiter=";")
test_data = np.reshape(test[:, 0:6], (len(test), 6))
test_labels = np.reshape(test[:, 6], (len(test), 1))


mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


main_input = Input(shape=(6,), name='main_input')

encoder = Dense(24, activation='relu')(main_input)
encoder = Dense(24, activation='relu')(encoder)
encoder = Dense(4, name='encoder_output')(encoder)

decoder = Dense(24, activation='relu', name='decoder_layer_0')(encoder)
decoder = Dense(24, activation='relu', name='decoder_layer_1')(decoder)
decoder = Dense(6, name='decoder_output')(decoder)

regression = Dense(18, activation='relu')(encoder)
regression = Dense(36, activation='relu')(regression)
regression = Dense(36, activation='relu')(regression)
regression = Dense(18, activation='relu')(regression)
regression = Dense(1, name='regression_output')(regression)


model = Model(inputs=[main_input], outputs=[regression, decoder])
model.compile(optimizer='adam', loss='mse', metrics='mae')
model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=50)

model.evaluate(test_data, [test_labels, test_data])


encoder_model = Model(main_input, encoder)
regression_model = Model(main_input, regression)
decoder_input = Input(shape=(4,))
dec = model.get_layer('decoder_layer_0')(decoder_input)
dec = model.get_layer('decoder_layer_1')(dec)
dec = model.get_layer('decoder_output')(dec)
decoder_model = Model(decoder_input, dec)

encoder_model.save('encoder_model.h5')
regression_model.save('regression_model.h5')
decoder_model.save('decoder_model.h5')

encoder_predict = encoder_model.predict(test_data)
regression_predict = np.hstack((test_labels, regression_model.predict(test_data)))
decoder_predict = decoder_model.predict(encoder_predict) * std + mean
np.savetxt('encoded.csv', encoder_predict, delimiter=';')
np.savetxt('regression.csv', regression_predict, delimiter=';')
np.savetxt('decoded.csv', decoder_predict, delimiter=';')
