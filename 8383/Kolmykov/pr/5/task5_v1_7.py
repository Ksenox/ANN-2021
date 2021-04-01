# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model


train_data = np.genfromtxt('train.csv', delimiter=';')
test_data = np.genfromtxt('test.csv', delimiter=';')

train_x = np.reshape(train_data[:, :6], (len(train_data), 6))
train_y = np.reshape(train_data[:, 6], (len(train_data), 1))
test_x = np.reshape(test_data[:, :6], (len(test_data), 6))
test_y = np.reshape(test_data[:, 6], (len(test_data), 1))

train_encoded_x = np.asarray([[arr[0], arr[3], arr[4], arr[5]] for arr in train_x])
test_encoded_x = np.asarray([[arr[0], arr[3], arr[4], arr[5]] for arr in train_x])

main_input = Input(shape=(6,), name='main_input')

encoding_layer = Dense(16, activation='relu')(main_input)
encoding_layer = Dense(16, activation='relu')(encoding_layer)
encoding_layer = Dense(16, activation='relu')(encoding_layer)
encoding_output = Dense(4, name='encoding_output')(encoding_layer)

decoding_layer = Dense(64, activation='relu')(encoding_output)
decoding_layer = Dense(64, activation='relu')(decoding_layer)
decoding_layer = Dense(64, activation='relu')(decoding_layer)
decoding_output = Dense(6, name='decoding_output')(decoding_layer)

regression_layer = Dense(64, activation='relu')(encoding_output)
regression_layer = Dense(64, activation='relu')(regression_layer)
regression_layer = Dense(64, activation='relu')(regression_layer)
regression_output = Dense(1, name='regression_output')(regression_layer)

model = Model(inputs=[main_input], outputs=[regression_output, encoding_output, decoding_output])
model.compile(optimizer='rmsprop', loss='mse', metrics='mae')
model.fit([train_x], [train_y, train_encoded_x, train_x], epochs=200, batch_size=5, validation_split=0)

test = [[25, 0.84, 0.98, 2, -5, 5]]
print(model.predict(test))

regression_model = Model(inputs=[main_input], outputs=[regression_output])
# print(regression_model.predict(test))
regression_prediction = regression_model.predict(test_x)

encoding_model = Model(inputs=[main_input], outputs=[encoding_output])
# print(encoding_model.predict(test))
encoding_prediction = encoding_model.predict(test_x)

decoding_model = Model(inputs=[main_input], outputs=[decoding_output])
# print(decoding_model.predict(test))
decoding_prediction = decoding_model.predict(test_x)

regression_model.save('regression_model.h5')
encoding_model.save('encoding_model.h5')
decoding_model.save('decoding_model.h5')

np.savetxt('regression_results.csv', np.hstack((test_y, regression_prediction)), delimiter=';')
np.savetxt('encoding_results.csv', np.hstack((test_encoded_x, encoding_prediction)), delimiter=';')
np.savetxt('decoding_results.csv', np.hstack((test_x, decoding_prediction)), delimiter=';')
