# 4.4

import numpy as np
import pandas

from keras.layers import Input, Dense
from keras.models import Model, load_model

v = 4  # var (10 mod 7) + 1
n_of_neurons = 128
epochs = 100
encode_dim = 6
N = 10000
def f1(x,e):
    return np.cos(x) + e

def f2(x,e):
    return -x + e

def f3(x,e):
    return np.sin(x) * x + e

def f4(x,e):
    return np.sqrt(np.abs(x)) + e

def f5(x,e):
    return  x ** 2 + e

def f6(x,e):
    return -np.abs(x) + 4 + e

def f7(x,e):
    return x - (x ** 2) / 5 + e

'''
def get_data(n_of_samples=1000):
    x = np.random.normal(0, 10, n_of_samples)
    e = np.random.normal(0, .3, n_of_samples)
    data = np.ones((n_of_samples, 7))
    data[:, 0] = np.cos(X) + e
    data[:, 1] = -X + e
    data[:, 2] = np.sin(X) * X + e
    data[:, 3] = np.sqrt(np.abs(X)) + e
    data[:, 4] = X ** 2 + e
    data[:, 5] = -np.abs(X) + 4 + e
    data[:, 6] = X - (X ** 2) / 5 + e
    return data
'''

def split_data(data, split_frac=.8):
    n_split = int(data.shape[0] * split_frac)

    train_data = data[:n_split, [i for i in range(7) if i != v - 1]]
    train_labels = data[:n_split, v - 1]

    test_data = data[n_split:, [i for i in range(7) if i != v - 1]]
    test_labels = data[n_split:, v - 1]

    return train_data, train_labels, test_data, test_labels

def gen_data(n_train=N, n_test = 200):
    n = n_train + n_test
    x = np.random.normal(0, 10, n)
    e = np.random.normal(0, .3, n)
    data = np.asarray([f1(x, e), f2(x, e), f3(x, e), f4(x, e), f5(x, e), f6(x, e)]).transpose()
    labels = np.asarray(f7(x, e))
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    test_data = data[n_train:]
    test_labels = labels[n_train:]
    np.savetxt("train_data.csv", train_data, delimiter=',')
    np.savetxt("train_labels.csv", train_labels, delimiter=',')
    np.savetxt("test_data.csv", test_data, delimiter=',')
    np.savetxt("test_labels.csv", test_labels, delimiter=',')


def load_data():
    train_data = np.genfromtxt('train_data.csv', delimiter=',')
    test_data = np.genfromtxt('test_data.csv', delimiter=',')
    train_labels = np.genfromtxt('train_labels.csv', delimiter=',')
    test_labels = np.genfromtxt('test_labels.csv', delimiter=',')

    return train_data, train_labels, test_data, test_labels


def get_encoder(input):
    hiden_encode_layer = Dense(n_of_neurons, activation='relu')(input)
    hiden_encode_layer = Dense(int(n_of_neurons/2), activation='relu')(hiden_encode_layer)
    encoder_output =  Dense(int(encode_dim), activation='relu', name='encoder_output')(hiden_encode_layer)
    return encoder_output

def get_decoder(input):
    hiden_decode_layer = Dense(n_of_neurons, activation='relu', name="decoder_1")(input)
    hiden_decode_layer = Dense(n_of_neurons, activation='relu', name="decoder_2")(hiden_decode_layer)
    decoder_output = Dense(encode_dim, activation='relu', name='decoder_output')(hiden_decode_layer)
    return decoder_output

def get_reg(input):
    hiden_reg_layer = Dense(n_of_neurons, activation='relu')(input)
    hiden_reg_layer = Dense(n_of_neurons, activation='relu')(hiden_reg_layer)
    reg_output = Dense(1, name='reg_output')(hiden_reg_layer)
    return reg_output


def build_and_fit():

    gen_data()
    train_data, train_labels, test_data, test_labels = load_data()

    main_input = Input(shape=(6,), name='main_input')

    encoder = get_encoder(main_input)

    decoder = get_decoder(encoder)

    reg = get_reg(encoder)


    full_model = Model(inputs=main_input, outputs=[reg, decoder])
    full_model.compile(loss='mse', optimizer="Adam",metrics=['mae'])

    full_model.fit(train_data, [train_labels, train_data], epochs=epochs, batch_size=16, validation_split=0.1, verbose=0)

    full_model.evaluate(test_data, [test_labels, test_data])

    encoder_model = Model(inputs=main_input, outputs=encoder)
    reg_model = Model(inputs=main_input, outputs=reg)
    decoder_model = Model(inputs=main_input, outputs=decoder)

    encoder_model.save('encoder_model.h5')
    reg_model.save('regression_model.h5')
    decoder_model.save('decoder_model.h5')

    encoder_model = load_model('encoder_model.h5', compile=False)
    reg_model = load_model('regression_model.h5', compile=False)
    decoder_model = load_model('decoder_model.h5', compile=False)

    encoded_data = encoder_model.predict(test_data)
    np.savetxt("encoded_data.csv", encoded_data, delimiter=',')

    decoded_data = decoder_model.predict(test_data)
    np.savetxt("decoded_data.csv", decoded_data, delimiter=',')

    regression_predictions = reg_model.predict(test_data)
    np.savetxt("reg_data.csv", regression_predictions, delimiter=',')



if __name__ == '__main__':
    build_and_fit()
    print("Done")