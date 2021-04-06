import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from dataset_generation import *

# var. 4, priznak 6

train_data = np.loadtxt("my_dataset_train.csv", delimiter='\t', dtype=np.float)
train_X = np.reshape(train_data[:, :6], (len(train_data), 6))
train_Y = np.reshape(train_data[:, 6], (len(train_data), 1))

test_data = np.loadtxt("my_dataset_test.csv", delimiter='\t', dtype=np.float)
test_X = np.reshape(test_data[:, :6], (len(test_data), 6))
test_Y = np.reshape(test_data[:, 6], (len(test_data), 1))

n_inputs = 6
# define encoder
visible = Input(shape=(n_inputs,))
e = Dense(n_inputs, activation='relu', name='encoder')(visible)
# define decoder
d = Dense(n_inputs, name='decoder')(e)
# output layer
l1 = Dense(64, activation='relu')(e)
l2 = Dense(32, activation='relu')(l1)
output = Dense(1, activation='linear', name='output')(l2)
# define autoencoder model
model = Model(inputs=visible, outputs=[output, d])
# compile autoencoder model
model.compile(optimizer='adam', loss='mse', metrics='mae')

model.fit(train_X, [train_Y, train_X], epochs=250, batch_size=5)

e_model = Model(inputs=visible, outputs=e)
d_model = Model(inputs=visible, outputs=d)
reg_model = Model(inputs=visible, outputs=output)

np.savetxt('encoded_model_prediction', e_model.predict(test_X), delimiter='\t')
np.savetxt('decoded_model_prediction', d_model.predict(e_model.predict(test_X)), delimiter='\t')
np.savetxt('regression_model_prediction', np.hstack((test_Y, reg_model.predict(test_X))), delimiter='\t')

e_model.save('encoded_model.h5')
d_model.save('decoded_model.h5')
reg_model.save('regression_model.h5')
