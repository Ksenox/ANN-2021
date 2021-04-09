import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


### 0
def generate(length):
    data = np.zeros([length, 6])
    X = np.random.normal(0, 10, length)
    e = np.random.normal(0, 0.3, length)

    data[:, 0] = np.array(X ** 2 + X + e)
    data[:, 1] = np.array(np.fabs(X) + e)
    data[:, 2] = np.array(np.sin(X - np.pi / 4) + e)
    data[:, 3] = np.array(np.log(np.fabs(X)) + e)
    data[:, 4] = np.array(-(X ** 3) + e)
    data[:, 5] = np.array(-X/4 + e)

    label = -X + e

    return data, label


### 1
train_data, train_labels = generate(1000)
test_data, test_labels = generate(100)

np.savetxt("1train_data.csv", train_data, delimiter=";")
np.savetxt("1train_labels.csv", train_labels, delimiter=";")
np.savetxt("1test_data.csv", test_data, delimiter=";")
np.savetxt("1test_labels.csv", test_labels, delimiter=";")

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

### 2
inputs = Input(shape=(6,))

encoded = Dense(30, activation='relu')(inputs)
encoded = Dense(20, activation='relu')(encoded)
encoded = Dense(4)(encoded)

decoded = Dense(20, activation='relu', name="dl_1")(encoded)
decoded = Dense(30, activation='relu', name="dl_2")(decoded)
decoded = Dense(6, name="decoder")(decoded)

regression = Dense(40, activation='relu')(encoded)
regression = Dense(50, activation='relu')(regression)
regression = Dense(1, name="regression")(regression)

### 3
model = Model(inputs, outputs=[regression, decoded])
model.compile(optimizer='adam', loss='mse', metrics=["mae"])

model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=50, verbose=0)

model.evaluate(test_data, [test_labels, test_data])

### 4
encoding_m = Model(inputs, encoded)
regression_m = Model(inputs, regression)

encoded_input = Input(shape=(4,))
decoding_layer = model.get_layer("dl_1")(encoded_input)
decoding_layer = model.get_layer("dl_2")(decoding_layer)
decoding_layer = model.get_layer("decoder")(decoding_layer)
decoding_m = Model(encoded_input, decoding_layer)

encoding_m.save('2encoding_m.h5')
regression_m.save('2regression_m.h5')
decoding_m.save('2decoding_mo.h5')

encoded_data = encoding_m.predict(test_data)

decoded_data = decoding_m.predict(encoded_data)
decoded_data *= std
decoded_data += mean
regression_predictions = regression_m.predict(test_data)
regression_predictions = regression_predictions.reshape((regression_predictions.shape[0]))
combined_data = np.asarray([test_labels, regression_predictions]).transpose()

np.savetxt("3decoded.csv", decoded_data, delimiter=';')
np.savetxt("3encoded.csv", encoded_data, delimiter=';')
np.savetxt("3regression.csv", combined_data, delimiter=';')
