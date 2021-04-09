import numpy as np
import math
import collections
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense

class pr5:
    def __init__(self, size = 300):
        self.functions = [self.__getattribute__(fun) for fun in dir(self) if fun.startswith('fun_')]
        self.data_x, self.data_y = self.genData(size)
        self.write_csv('./data_x.csv', self.data_x)
        self.write_csv('./data_y.csv', self.data_x)
        self.test_data_x, self.test_data_y = self.genData(size//5)
        self.dataset_size = size
        self.test_dataset_size = size/5
        self.dataset_shape = 6

    def genData(self, size):
        x_data = []
        y_data = []
        for i in range(size):
            X = np.random.normal(3, 10)
            e = np.random.normal(0, 0.3)
            data = [fun(X, e) for fun in self.functions]
            y = self.aim_fun(X, e)
            x_data.append(data)
            y_data.append(y)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        x_mean = x_data.mean(axis=0)
        x_std = x_data.std(axis=0)
        x_data -= x_mean
        x_data /= x_std
        y_mean = y_data.mean(axis=0)
        y_std = y_data.std(axis=0)
        y_data -= y_mean
        y_data /= y_std
        return np.round(x_data, decimals=3), np.round(y_data, decimals=3)

    def write_csv(self, path, data):
        with open(path, 'w', newline='') as file:
            my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if isinstance(data, collections.Iterable) and isinstance(data[0], collections.Iterable):
                for i in data:
                    my_csv.writerow(i)
            else:
                my_csv.writerow(data)

    def fun_1(self, X, e):
        return X**2 + e

    def fun_2(self, X, e):
        return math.sin(X/2)+e

    def aim_fun(self, X, e):
        return math.cos(2*X)+e

    def fun_4(self, X, e):
        return X-3+e

    def fun_5(self, X, e):
        return -X+e

    def fun_6(self, X, e):
        return math.fabs(X)+e

    def fun_7(self, X, e):
        return (X**3)/4 + e

    def create_models(self):
        input = Input(shape=(6,))
        encoded = Dense(32, activation='tanh')(input)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(32, activation='tanh')(encoded)
        encoded = Dense(6, activation='tanh')(encoded)

        decoded = Dense(6, activation='tanh')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(32, activation='tanh')(decoded)
        decoded = Dense(6, activation='linear', name='decoder_output')(decoded)

        regr = Dense(64, activation='relu')(encoded)
        regr = Dense(64, activation='tanh')(regr)
        regr = Dense(64, activation='tanh')(regr)
        regr = Dense(64, activation='tanh')(regr)
        regr = Dense(32, activation='tanh')(regr)
        regr = Dense(1, activation='linear', name="regr_output")(regr)

        regr = Model(input, outputs = [regr, decoded], name="regr")
        encoded = Model(input, encoded, name="encoder")

        input_encoded_data = Input(shape=(6,))
        decoded = Dense(32, activation='tanh')(input_encoded_data)
        decoded = Dense(6, activation='tanh')(decoded)
        decoded = Model(input_encoded_data, decoded, name="decoder")

        return encoded, decoded, regr, input

    def start(self):
        encoder, decoder, regr, input_layer = self.create_models()
        regr.compile(optimizer="adam", loss="mse", metrics=['mae'])
        epochs = 60
        history = regr.fit(self.data_x, {'regr_output':self.data_y, 'decoder_output':self.data_x}, epochs=epochs, batch_size=6, verbose=1, validation_data=(self.test_data_x, [self.test_data_y, self.test_data_x]))
        self.plot(history, epochs)

        encoder.save('encoder.h5')
        decoder.save('decoder.h5')
        regr.save('regr.h5')
        encoded_data = encoder.predict(self.test_data_x)
        decoded_data = decoder.predict(encoded_data)
        regr_data = regr.predict(self.test_data_x)
        self.write_csv('ecoded_data.csv', encoded_data)
        self.write_csv('decoded_data.csv', decoded_data)
        self.write_csv('regr_pred.csv', regr_data[0])
        self.write_csv('regr_res.csv', self.test_data_y)

    def plot(self, history, epochs):
        loss = history.history['loss']
        v_loss = history.history['val_loss']
        x = range(1, epochs+1)
        plt.plot(x, loss, 'b', label='train')
        plt.plot(x, v_loss, 'r', label='validation')
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
        plt.clf()


practice = pr5(500)
practice.start()
