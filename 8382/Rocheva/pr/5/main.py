import csv
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def generate_data(size):
    data = []
    target = []
    for i in range(size):
        X = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        data.append((np.cos(X)+e, -X+e, np.sin(X)*X+e, np.sqrt(np.fabs(X))+e, -np.fabs(X)+4, X-(X**2)/5+e))
        target.append([X ** 2 + e])
    return np.round(np.array(data), decimals=5), np.round(np.array(target), decimals=5)


def write_csv(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for item in data:
            writer.writerow(item)


train_data, train_targets = generate_data(500)
test_data, test_targets = generate_data(50)
write_csv('train_data.csv', np.round(train_data, 5))
write_csv('train_targets.csv', np.round(train_targets, 5))
write_csv('test_data.csv', np.round(test_data, 5))
write_csv('test_targets.csv', np.round(test_targets, 5))


mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std
test_data -= mean
test_data /= std

inputs = Input(shape=(6,))

encoded = Dense(32, activation='relu')(inputs)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(2, activation='relu')(encoded)

decoded = Dense(64, activation='relu', name='dec1')(encoded)
decoded = Dense(64, activation='relu', name='dec2')(decoded)
decoded = Dense(64, activation='relu', name='dec3')(decoded)
decoded = Dense(6, name='dec4')(decoded)

regress = Dense(32, activation='relu')(encoded)
regress = Dense(64, activation='relu')(regress)
regress = Dense(64, activation='relu')(regress)
regress = Dense(64, activation='relu')(regress)
regress = Dense(1, name='regress')(regress)

encoder = Model(inputs, encoded)
regress_model = Model(inputs, regress)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
autoencoder.fit(train_data, train_data, epochs=150, batch_size=5, verbose=0, validation_data=(test_data, test_data))

encode_data = encoder.predict(test_data)

input_dec = Input(shape=(2,))

decoder = autoencoder.get_layer('dec1')(input_dec)
decoder = autoencoder.get_layer('dec2')(decoder)
decoder = autoencoder.get_layer('dec3')(decoder)
decoder = autoencoder.get_layer('dec4')(decoder)
decoder = Model(input_dec, decoder)
decode_data = decoder.predict(encode_data)

regress_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
regress_model.fit(train_data, train_targets, epochs=150, batch_size=5, verbose=0, validation_data=(test_data, test_targets))
regress_data = regress_model.predict(test_data)

decoder.save('decoder.h5')
encoder.save('encoder.h5')
regress_model.save('regression.h5')

write_csv('encoded_data.csv', np.round(encode_data, 5))
write_csv('decoded_data.csv', np.round(decode_data, 5))
write_csv('regress_data.csv', np.round(regress_data, 5))
