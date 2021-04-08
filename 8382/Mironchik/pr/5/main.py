import numpy as np
from tensorflow import keras
from keras import layers


def f1(x, e):
    return x ** 2 + e


def f2(x, e):
    return np.sin(x / 2) + e


def f3(x, e):
    return np.cos(2 * x) + e


def f4(x, e):
    return x - 3 + e


def f5(x, e):
    return -x + e


def f6(x, e):
    return np.abs(x) + e


def f7(x, e):
    return x ** 3 / 4 + e


def generate(train_size=10000, test_size=100):
    total_size = train_size + test_size
    x = np.random.normal(3, 10, total_size)
    e = np.random.normal(0, 0.3, total_size)
    data = np.asarray([f1(x, e), f2(x, e), f3(x, e), f4(x, e), f5(x, e), f6(x, e)]).transpose()
    labels = np.asarray(f7(x, e))
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    test_data = data[train_size:]
    test_labels = labels[train_size:]
    np.savetxt("train_data.csv", train_data, delimiter=';')
    np.savetxt("train_labels.csv", train_labels, delimiter=';')
    np.savetxt("test_data.csv", test_data, delimiter=';')
    np.savetxt("test_labels.csv", test_labels, delimiter=';')


generate()

train_data = np.genfromtxt('train_data.csv', delimiter=';')
test_data = np.genfromtxt('test_data.csv', delimiter=';')
train_labels = np.genfromtxt('train_labels.csv', delimiter=';')
test_labels = np.genfromtxt('test_labels.csv', delimiter=';')

print(np.mean(train_labels))

inputs = keras.Input((6,))

encoder = keras.layers.Dense(32, activation='relu')(inputs)
encoder = keras.layers.Dense(32, activation='relu')(encoder)
encoder = keras.layers.Dense(2)(encoder)

decoder = keras.layers.Dense(32, activation='relu', name='decoder1')(encoder)
decoder = keras.layers.Dense(32, activation='relu', name='decoder2')(decoder)
decoder = keras.layers.Dense(6, name='decoder_out')(decoder)

regression = keras.layers.Dense(32, activation='relu')(encoder)
regression = keras.layers.Dense(32, activation='relu')(regression)
regression = keras.layers.Dense(1, name='regression_out')(regression)

model = keras.Model(inputs, outputs=[regression, decoder])
model.compile(optimizer='adam', loss='mse', metrics=["mae"], loss_weights=[1, 0.9])

model.fit(train_data, [train_labels, train_data], epochs=100, batch_size=100, validation_split=0.1, verbose=0)

model.evaluate(test_data, [test_labels, test_data])

encoder_model = keras.Model(inputs, encoder)
regression_model = keras.Model(inputs, regression)

encoded_input = keras.Input(shape=(2,))
decoder_layer = model.get_layer("decoder1")(encoded_input)
decoder_layer = model.get_layer("decoder2")(decoder_layer)
decoder_layer = model.get_layer("decoder_out")(decoder_layer)
decoder_model = keras.Model(encoded_input, decoder_layer)

encoder_model.save('encoder_model.h5')
regression_model.save('regression_model.h5')
decoder_model.save('decoder_model.h5')

encoder_model = keras.models.load_model('encoder_model.h5', compile=False)
regression_model = keras.models.load_model('regression_model.h5', compile=False)
decoder_model = keras.models.load_model('decoder_model.h5', compile=False)

encoded_data = encoder_model.predict(test_data)
np.savetxt("encoded_test_data.csv", encoded_data, delimiter=';')

decoded_data = decoder_model.predict(encoded_data)
np.savetxt("decoded_test_data.csv", decoded_data, delimiter=';')

regression_predictions = regression_model.predict(test_data)
regression_predictions = regression_predictions.reshape((regression_predictions.shape[0]))
combined_data = np.asarray([test_labels, regression_predictions]).transpose()
np.savetxt("regression_test_predictions.csv", combined_data, delimiter=';')
