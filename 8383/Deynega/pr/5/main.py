import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

train_size = 500
test_size = 50

def gen_data(size):
    data = np.empty([size, 6])
    labels = np.empty(size)
    for i in range(size):
        x = np.random.normal(-5, 10)
        e = np.random.normal(0, 0.3)
        data[i] = np.array([-1 * x ** 3, np.sin(3*x), np.exp(x),
                            x+4, -1 * x + np.sqrt(np.fabs(x)), x])
        data[i] += e
        labels[i] = np.log(np.fabs(x)) + e
    return data, labels



train_data, train_labels = gen_data(train_size)
test_data, test_labels = gen_data(test_size)

np.savetxt("train_data.csv", train_data, delimiter=";")
np.savetxt("train_labels.csv", train_labels, delimiter=";")
np.savetxt("test_data.csv", test_data, delimiter=";")
np.savetxt("test_labels.csv", test_labels, delimiter=";")

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

np.savetxt("train_data_n.csv", train_data, delimiter=";")
np.savetxt("test_data_n.csv", test_data, delimiter=";")


inputs = Input(shape=(6,))

encoder_1 = Dense(64, activation='relu')(inputs)
encoder_2 = Dense(32, activation='relu')(encoder_1)
encoder_3 = Dense(4, activation='relu', name='encoder_output')(encoder_2)

decoder_1 = Dense(32, activation='relu', name='decoder_1')(encoder_3)
decoder_2 = Dense(64, activation='relu', name='decoder_2')(decoder_1)
decoder_3 = Dense(6, name='decoder_3')(decoder_2)

regr_1 = Dense(32, activation='relu')(encoder_3)
regr_2 = Dense(64, activation='relu')(regr_1)
regr_3 = Dense(45, activation='relu')(regr_2)
regression = Dense(1, name='predicted_output')(regr_3)

model = Model(inputs=inputs, outputs=[decoder_3, regression])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_data, [train_data, train_labels], epochs=200, batch_size=10)
model.save('main_model.h5')

encoder_model = Model(inputs=inputs, outputs=encoder_3)
encoder_pred = encoder_model.predict(test_data)
np.savetxt("encoded_test_n_data.csv", encoder_pred, delimiter=";")
model.save('encoder_model.h5')

regr_model = Model(inputs=inputs, outputs=regression)
regression_pred = regr_model.predict(test_data)
np.savetxt("regression.csv", regression_pred, delimiter=";")
model.save('regression_model.h5')

decoder_inputs = Input(shape=(4,))
dec_1 = model.get_layer('decoder_1')(decoder_inputs)
dec_2 = model.get_layer('decoder_2')(dec_1)
dec_3 = model.get_layer('decoder_3')(dec_2)
decoder_model = Model(decoder_inputs, dec_3)
decoder_prediction = decoder_model.predict(encoder_pred)
np.savetxt("decoded_test_n_data.csv", decoder_prediction, delimiter=";")
model.save('decoder_model.h5')

print(decoder_prediction, "\n\n")
print(test_data, "\n----------------------------------------\n")
print(regression_pred, "\n\n")
print(test_labels, "\n\n")
