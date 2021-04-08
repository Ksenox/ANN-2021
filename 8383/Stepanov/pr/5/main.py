import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

train_dataset = np.fromfile('train_dataset.csv', dtype ='float', sep =' ').reshape(-1, 7)
test_dataset = np.fromfile('test_dataset.csv', dtype ='float', sep =' ').reshape(-1, 7)

train_data = train_dataset[:, 0:6].reshape(-1, 6)
train_targets = train_dataset[:, 6].reshape(-1, 1)

test_data = test_dataset[:, 0:6].reshape(-1, 6)
test_targets = test_dataset[:, 6].reshape(-1, 1)

inp = Input(shape=(6,))

# encoder
dense_ecoder = Dense(64, activation='relu', name='dense_ecoder')(inp)
dense_ecoder = Dense(32, activation='relu')(dense_ecoder)
dense_ecoder = Dense(4)(dense_ecoder)

# regression
dense_1 = Dense(32, activation='relu')(dense_ecoder)
dense_2 = Dense(64, activation='relu')(dense_1)
out_reg = Dense(1, name='out_reg')(dense_2)

# decoder
dense_decoder = Dense(32, activation='relu', name='dense_decoder_1')(dense_ecoder)
dense_decoder = Dense(64, activation='relu', name='dense_decoder_2')(dense_decoder)
dense_decoder = Dense(6, name='dense_decoder_out')(dense_decoder)

# configuration
model = Model(inputs=inp, outputs=[out_reg, dense_decoder])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(train_data, [train_targets, train_data], epochs=200, batch_size=10)

# encoder model
ecoder_model = Model(inputs=inp, outputs=dense_ecoder)
encoder_pred = np.asarray(ecoder_model.predict(test_data))
np.savetxt('outOfModels/data_encoder_model.csv', encoder_pred)
ecoder_model.save('models/encoder_model.h5')

# regression model
reg_model = Model(inputs=inp, outputs=out_reg)
reg_pred = np.asarray(reg_model.predict(test_data))
np.savetxt('outOfModels/data_reg_model.csv', np.hstack((test_targets, reg_pred)))
ecoder_model.save('models/reg_model.h5')

# decoder model
decoder_input = Input(shape=(4,))
decoder_dens = model.get_layer('dense_decoder_1')(decoder_input)
decoder_dens = model.get_layer('dense_decoder_2')(decoder_dens)
decoder_dens = model.get_layer('dense_decoder_out')(decoder_dens)
decoder_model = Model(inputs=decoder_input, outputs=decoder_dens)
decoder_pred = np.asarray(decoder_model.predict(encoder_pred))
np.savetxt('outOfModels/data_decoder_model.csv', decoder_pred)
ecoder_model.save('models/decoder_model.h5')

print('end')